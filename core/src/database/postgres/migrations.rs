use crate::database::generate::{
    generate_indexer_contract_schema_name, generate_table_full_name,
    GenerateTablesForIndexerSqlError,
};
use crate::helpers::camel_to_snake;
use crate::indexer::Indexer;
use crate::{ABIItem, PostgresClient};
use std::path::Path;

/// Atomically incrementing version number for migrations.
///
/// Note: if all migrations are idempotent on re-run we can avoid tracking last-known migration
/// entirely, however the cost associated with checking already run migrations is low enough that
/// we can design it to be future-proof at least until are clear it's not worth it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Migration {
    /// Add `block_timestamp` column to all tables, nullable by default.
    V1,
    /// Rewrite reorg journal keys of the indexer's custom tables from the quoted format
    /// (`pool_address='0x..'`) to the unquoted one the reorg recalculation matches.
    V2,
}

impl Migration {
    pub const ALL: &'static [Migration] = &[Migration::V1, Migration::V2];

    pub fn as_u32(&self) -> u32 {
        match self {
            Migration::V1 => 1,
            Migration::V2 => 2,
        }
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Migration::V1),
            2 => Some(Migration::V2),
            _ => None,
        }
    }

    /// SQL for one raw event table; empty for migrations that do not touch event tables.
    pub fn generate_table_migration_sql(&self, table_name: &str) -> String {
        match self {
            Migration::V1 => {
                format!("ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS block_timestamp TIMESTAMPTZ;")
            }
            Migration::V2 => String::new(),
        }
    }

    /// SQL for one custom (derived) table; empty for migrations that do not touch them.
    pub fn generate_derived_table_migration_sql(&self, derived_table: &str) -> String {
        match self {
            Migration::V1 => String::new(),
            Migration::V2 => unquote_journal_where_keys_sql(derived_table),
        }
    }
}

/// Matches one quoted `col='value'` segment of an old-format `where_key`; `bytes` values
/// (`'\x..'`) are excluded because the writer still journals them quoted.
const QUOTED_WHERE_SEGMENT: &str = r"(^|,)([A-Za-z0-9_]+)='(?!\\x)((?:[^']|'')*)'(?=,|$)";

/// Strips the SQL quotes the journal writer used to put around `address` and `string`
/// values in `where_key` and undoes their `''` escaping, e.g.
/// `label='o''neil',pool_address='0x..'` -> `label=o'neil,pool_address=0x..`.
///
/// Only rows with a quoted segment are touched. In such a row every string value is
/// quoted, so a `''` inside it is always an escape. The migration runs once per indexer,
/// before indexing starts, so none of these tables has a new-format row yet, and it is
/// scoped to the indexer's own tables because `derived_op_log` is shared.
fn unquote_journal_where_keys_sql(derived_table: &str) -> String {
    let quote = |s: &str| s.replace('\'', "''");
    let pattern = quote(QUOTED_WHERE_SEGMENT);
    format!(
        "UPDATE rindexer_internal.derived_op_log \
         SET where_key = replace(regexp_replace(where_key, '{pattern}', '\\1\\2=\\3', 'g'), '''''', '''') \
         WHERE derived_table = '{table}' AND where_key ~ '{pattern}';",
        table = quote(derived_table),
    )
}

/// Execute any unapplied migrations for rindexer project.
///
/// ## Migration flow
///
/// 1. Check any already applied migrations
/// 2. Get the migrations that have not yet been applied
/// 3. Generate the migration SQL and apply them
/// 4. Insert new rows reflecting that these migrations have been applied
pub async fn execute_migrations_for_indexer_sql(
    client: &PostgresClient,
    project_path: &Path,
    indexer: &Indexer,
    disable_event_tables: bool,
) -> Result<Vec<Migration>, GenerateTablesForIndexerSqlError> {
    if disable_event_tables {
        return Ok(vec![]);
    }

    let applied = client
        .query(
            &format!(
                r#"
                 SELECT version FROM rindexer_internal.{indexer_name}_last_run_migrations_sql
                 WHERE migration_applied IS TRUE
                "#,
                indexer_name = camel_to_snake(&indexer.name)
            ),
            &[],
        )
        .await?;

    let applied = applied
        .iter()
        .filter_map(|row| {
            let id: i32 = row.get(0);
            Migration::from_u32(id as u32)
        })
        .collect::<Vec<_>>();
    let unapplied =
        Migration::ALL.iter().cloned().filter(|v| !applied.contains(v)).collect::<Vec<_>>();

    let mut statements = String::new();

    for version in &unapplied {
        for contract in &indexer.contracts {
            let contract_name = contract.before_modify_name_if_filter_readonly();
            let abi_items = ABIItem::read_abi_items(project_path, contract)?;
            let events = ABIItem::extract_event_names_and_signatures_from_abi(abi_items)?;
            let schema_name = generate_indexer_contract_schema_name(&indexer.name, &contract_name);

            // Run migrations on every raw event table that exists — the same
            // predicate DDL creation uses (include_events plus table events)
            let raw_events: Vec<_> =
                events.iter().filter(|e| contract.stores_raw_events(&e.name)).collect();

            for event_info in raw_events {
                let table_name = format!("{}.{}", schema_name, camel_to_snake(&event_info.name));
                statements.push_str(&version.generate_table_migration_sql(&table_name));
            }

            for table in contract.tables.iter().flatten() {
                let derived_table = generate_table_full_name(
                    &indexer.name,
                    &contract_name,
                    &table.name,
                    table.database.as_deref(),
                );
                statements.push_str(&version.generate_derived_table_migration_sql(&derived_table));
            }
        }
    }

    client.batch_execute(&statements).await?;

    client
        .query(
            &format!(
                r#"
                 INSERT INTO rindexer_internal.{indexer_name}_last_run_migrations_sql (version, migration_applied)
                 SELECT UNNEST($1::INT[]), TRUE
                 ON CONFLICT (version) DO NOTHING
                "#,
                indexer_name = camel_to_snake(&indexer.name),
            ),
            &[&unapplied.iter().map(|v| v.as_u32() as i32).collect::<Vec<_>>()],
        )
        .await?;

    Ok(unapplied)
}
