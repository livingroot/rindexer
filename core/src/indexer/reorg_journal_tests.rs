//! Docker-backed regression for the Postgres reorg journal of `set`/`max`/`min` columns.
//!
//! Every step goes through production code: the tables come from the real DDL generator,
//! the rows and their journal entries from [`process_table_operations`] (the eager no-code
//! writer), the reorg plan from [`build_derived_tables_for_event`], and the recalculation
//! from [`ReorgTask`]. The e2e journal tests in `core/tests/e2e_reorg.rs` insert
//! `derived_op_log` rows by hand, so they could not notice the writer and the
//! recalculation disagreeing on the `where_key` format, which is what this module pins.
//!
//! The `where` has an `address` and a `string` column, and one string carries an
//! apostrophe, the one character the writer escapes.
//!
//! Not covered, on purpose:
//! - `bytes` keys. The journal still stores them quoted (`col='\x..'`) while the
//!   recalculation rebuilds `col::TEXT`, so a `bytes` column in `where` never matches its
//!   journal entries. That is a separately disclosed limitation.
//! - ClickHouse. Its recalculation uses a correlated subquery ClickHouse rejects, and its
//!   writer does not apply `max`/`min`; both are known limitations outside this fix.
//!
//! Requires Docker (testcontainers). Tests set the process-global `DATABASE_URL`, so run
//! them under nextest, which gives each test its own process:
//!
//! ```text
//! cargo nextest run -p rindexer reorg_journal_tests
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use alloy::dyn_abi::DynSolValue;
use alloy::primitives::{Address, B256, U256};
use testcontainers::runners::AsyncRunner;
use testcontainers::ContainerAsync;
use testcontainers_modules::postgres::Postgres;

use super::reorg::task::ReorgTask;
use super::reorg::window::BlockChainWindow;
use super::start::build_derived_tables_for_event;
use super::tables::{process_table_operations, TableRuntime, TxMetadata};
use super::Indexer;
use crate::database::generate::generate_tables_for_indexer_sql;
use crate::database::postgres::client::{PostgresClient, TEST_DATABASE_URL_LOCK};
use crate::database::postgres::migrations::{execute_migrations_for_indexer_sql, Migration};
use crate::manifest::contract::Contract;
use crate::manifest::core::Constants;
use crate::manifest::native_transfer::NativeTransfers;
use crate::provider::ChainProvider;
use crate::types::core::LogParam;

const NETWORK: &str = "ethereum";
const INDEXER: &str = "journal_tests";
const CONTRACT: &str = "Pool";
const EVENT: &str = "Swap";
const POOL: &str = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640";
const QUOTED_LABEL: &str = "o'neil";
const PLAIN_LABEL: &str = "plain";

const ABI: &str = r#"[{"type":"event","name":"Swap","anonymous":false,"inputs":[
    {"name":"label","type":"string","indexed":false},
    {"name":"price","type":"uint256","indexed":false}]}]"#;

/// One row per (pool, label), with every journaled action on the same price.
const CONTRACT_YAML: &str = r#"
name: Pool
details:
  - network: ethereum
    address: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    start_block: "1"
abi: ./Pool.abi.json
tables:
  - name: pool_state
    columns:
      - name: pool_address
        type: address
      - name: label
        type: string
      - name: last_price
        type: uint256
        default: "0"
      - name: max_price
        type: uint256
        default: "0"
      - name: min_price
        type: uint256
        default: "0"
    events:
      - event: Swap
        operations:
          - type: upsert
            where:
              pool_address: $rindexer_contract_address
              label: $label
            set:
              - column: last_price
                action: set
                value: $price
              - column: max_price
                action: max
                value: $price
              - column: min_price
                action: min
                value: $price
"#;

const FIRST_BLOCK: u64 = 100;
const FORK_POINT: u64 = FIRST_BLOCK + 1;

/// Prices per label at blocks 100, 101, 102. The reorg drops 101 and 102, and every
/// journaled column then has a different value than before it: last 10 -> 20, max 30 -> 20,
/// min 10 -> 20 for the quoted label (and 5 -> 6, 7 -> 6, 5 -> 6 for the plain one).
const PRICES: [(&str, [u64; 3]); 2] = [(QUOTED_LABEL, [20, 30, 10]), (PLAIN_LABEL, [6, 7, 5])];

/// (last, max, min) per label.
type Prices = HashMap<String, (u64, u64, u64)>;

fn expected(blocks: usize) -> Prices {
    PRICES
        .iter()
        .map(|(label, prices)| {
            let kept = &prices[..blocks];
            let last = *kept.last().unwrap();
            let max = *kept.iter().max().unwrap();
            let min = *kept.iter().min().unwrap();
            (label.to_string(), (last, max, min))
        })
        .collect()
}

fn swap_events() -> Vec<(Vec<LogParam>, String, TxMetadata)> {
    let pool: Address = POOL.parse().unwrap();
    let mut events = Vec::new();
    for (block_offset, block) in (FIRST_BLOCK..FIRST_BLOCK + 3).enumerate() {
        for (log_index, (label, prices)) in PRICES.iter().enumerate() {
            let params = vec![
                LogParam::new("label".to_string(), DynSolValue::String(label.to_string())),
                LogParam::new(
                    "price".to_string(),
                    DynSolValue::Uint(U256::from(prices[block_offset]), 256),
                ),
            ];
            let metadata = TxMetadata {
                block_number: block,
                block_timestamp: None,
                tx_hash: B256::from(U256::from(block * 1_000 + log_index as u64)),
                block_hash: B256::from(U256::from(block)),
                contract_address: pool,
                log_index: U256::from(log_index),
                tx_index: log_index as u64,
            };
            events.push((params, NETWORK.to_string(), metadata));
        }
    }
    events
}

/// A Postgres container with the indexer's real schema (tables, journal, migrations
/// bookkeeping) and the table runtimes the writer takes.
struct Harness {
    _container: ContainerAsync<Postgres>,
    _project: tempfile::TempDir,
    pg: Arc<PostgresClient>,
    indexer: Indexer,
    tables: Vec<TableRuntime>,
}

impl Harness {
    async fn start() -> Self {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let container = Postgres::default().start().await.expect("failed to start postgres");
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let pg = {
            let _guard = TEST_DATABASE_URL_LOCK.lock().await;
            std::env::set_var(
                "DATABASE_URL",
                format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres?sslmode=disable"),
            );
            Arc::new(PostgresClient::new().await.expect("failed to create postgres client"))
        };

        // The project directory only has to hold the ABI the DDL generator reads.
        let project = tempfile::tempdir().unwrap();
        std::fs::write(project.path().join("Pool.abi.json"), ABI).unwrap();
        let contract: Contract = serde_yaml::from_str(CONTRACT_YAML).unwrap();
        let indexer = Indexer {
            name: INDEXER.to_string(),
            contracts: vec![contract.clone()],
            native_transfers: NativeTransfers::default(),
        };
        let ddl = generate_tables_for_indexer_sql(project.path(), &indexer, false).unwrap();
        pg.batch_execute(ddl.as_str()).await.expect("create postgres tables");

        let tables = contract
            .tables
            .unwrap()
            .into_iter()
            .map(|table| TableRuntime::new(table, INDEXER, CONTRACT))
            .collect();

        Self { _container: container, _project: project, pg, indexer, tables }
    }

    fn table(&self) -> &str {
        &self.tables[0].full_table_name
    }

    async fn write_swaps(&self) {
        let providers: Arc<HashMap<String, Arc<dyn ChainProvider>>> = Arc::new(HashMap::new());
        process_table_operations(
            &self.tables,
            EVENT,
            &swap_events(),
            Some(self.pg.clone()),
            None,
            providers,
            &Constants::new(),
            &HashMap::new(),
            None,
        )
        .await
        .expect("the writer must apply all swaps");
    }

    /// Runs the reorg the way the indexer would plan it, dropping blocks 101 and 102.
    async fn reorg(&self) {
        let mut derived_tables = HashMap::new();
        let mut rollback_plans = HashMap::new();
        build_derived_tables_for_event(
            EVENT,
            INDEXER,
            CONTRACT,
            NETWORK,
            &HashMap::new(),
            &self.tables,
            &mut derived_tables,
            &mut rollback_plans,
        )
        .expect("build the reorg plan");

        let task = ReorgTask {
            network: NETWORK.to_string(),
            fork_point: FORK_POINT,
            detection_point: FIRST_BLOCK + 2,
            event_tables: vec![],
            derived_tables: derived_tables.remove(NETWORK).unwrap(),
            canonical_blocks: vec![],
        };
        let rollback_plan = rollback_plans.remove(NETWORK).unwrap_or_default();
        task.execute_with_rollback_plan(
            &mut BlockChainWindow::try_new(256).unwrap(),
            Some(&self.pg),
            None,
            None,
            &rollback_plan,
        )
        .await
        .unwrap_or_else(|e| panic!("reorg task: {e:#}"));
    }

    async fn prices(&self) -> Prices {
        let rows = self
            .pg
            .query(
                &format!(
                    "SELECT label, last_price::TEXT, max_price::TEXT, min_price::TEXT FROM {} \
                     WHERE network = $1 AND pool_address = $2",
                    self.table()
                ),
                &[&NETWORK, &POOL],
            )
            .await
            .expect("read prices");
        rows.iter()
            .map(|row| {
                let number = |i: usize| row.get::<_, String>(i).parse::<u64>().unwrap();
                (row.get::<_, String>(0), (number(1), number(2), number(3)))
            })
            .collect()
    }

    async fn journal_keys(&self) -> Vec<String> {
        let rows = self
            .pg
            .query(
                "SELECT DISTINCT where_key FROM rindexer_internal.derived_op_log \
                 WHERE derived_table = $1 ORDER BY where_key",
                &[&self.table()],
            )
            .await
            .expect("read journal keys");
        rows.iter().map(|row| row.get(0)).collect()
    }

    async fn insert_journal(&self, where_key: &str, column: &str, value: u64, block: u64) {
        self.pg
            .execute(
                "INSERT INTO rindexer_internal.derived_op_log \
                 (derived_table, network, where_key, column_name, value, block_number, tx_index, log_index) \
                 VALUES ($1, $2, $3, $4, $5, $6, 0, 0)",
                &[
                    &self.table(),
                    &NETWORK,
                    &where_key,
                    &column,
                    &rust_decimal::Decimal::from(value),
                    &(block as i64),
                ],
            )
            .await
            .expect("insert journal row");
    }
}

#[tokio::test]
async fn journal_written_by_the_writer_drives_reorg_recalculation() {
    let harness = Harness::start().await;
    harness.write_swaps().await;
    assert_eq!(harness.prices().await, expected(3), "before the reorg");

    // The journal holds the values unquoted, the way the recalculation rebuilds them.
    assert_eq!(
        harness.journal_keys().await,
        vec![
            format!("label={QUOTED_LABEL},pool_address={POOL}"),
            format!("label={PLAIN_LABEL},pool_address={POOL}"),
        ]
    );

    harness.reorg().await;
    assert_eq!(harness.prices().await, expected(1), "only block 100 survives the reorg");
}

/// Journal rows written before the fix keep the quoted format. Without the V2 migration
/// they would never match again, and a `max`/`min` recalculated only from newer rows
/// would lose the history they hold: here the all-time maximum of 1000 at block 50.
#[tokio::test]
async fn migration_rewrites_legacy_journal_keys() {
    let harness = Harness::start().await;

    let legacy_quoted = format!("label='o''neil',pool_address='{POOL}'");
    let legacy_plain = format!("label='{PLAIN_LABEL}',pool_address='{POOL}'");
    // A `bytes` key keeps its quotes: the writer still journals it that way.
    let legacy_bytes = "hash='\\xabcd'".to_string();
    for column in ["last_price", "max_price", "min_price"] {
        harness.insert_journal(&legacy_quoted, column, 1000, 50).await;
        harness.insert_journal(&legacy_plain, column, 1, 50).await;
        harness.insert_journal(&legacy_bytes, column, 1, 50).await;
    }

    let applied = execute_migrations_for_indexer_sql(
        &harness.pg,
        harness._project.path(),
        &harness.indexer,
        false,
    )
    .await
    .expect("migrations");
    assert!(applied.contains(&Migration::V2));

    let migrated_quoted = format!("label={QUOTED_LABEL},pool_address={POOL}");
    let migrated_plain = format!("label={PLAIN_LABEL},pool_address={POOL}");
    let mut keys = vec![legacy_bytes, migrated_quoted, migrated_plain];
    keys.sort();
    assert_eq!(harness.journal_keys().await, keys, "legacy keys rewritten, bytes untouched");

    harness.write_swaps().await;
    harness.reorg().await;

    // Block 50 (legacy, now migrated) and block 100 survive: last comes from block 100,
    // max and min fold both.
    let after: Prices = HashMap::from([
        (QUOTED_LABEL.to_string(), (20, 1000, 20)),
        (PLAIN_LABEL.to_string(), (6, 6, 1)),
    ]);
    assert_eq!(harness.prices().await, after);
}
