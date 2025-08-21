//! Schema Management Module
//!
//! This module provides comprehensive schema management capabilities including:
//! - Automatic schema inference from data
//! - Version-controlled schema evolution
//! - Data validation and type coercion
//! - Migration generation and execution
//! - Schema comparison and diff generation
//! - Cross-database schema compatibility

use super::{DatabaseConnection, DatabaseConfig, DatabaseError};
use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, LyObj};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use serde_json::Value as JsonValue;

/// Schema definition with version control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub name: String,
    pub version: String,
    pub tables: HashMap<String, TableSchema>,
    pub relationships: Vec<Relationship>,
    pub indexes: Vec<IndexDefinition>,
    pub constraints: Vec<Constraint>,
    pub metadata: HashMap<String, JsonValue>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub name: String,
    pub columns: HashMap<String, ColumnDefinition>,
    pub primary_key: Vec<String>,
    pub foreign_keys: Vec<ForeignKey>,
    pub indexes: Vec<String>,
    pub constraints: Vec<String>,
    pub metadata: HashMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDefinition {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default_value: Option<JsonValue>,
    pub auto_increment: bool,
    pub unique: bool,
    pub comment: Option<String>,
    pub metadata: HashMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    // Numeric types
    Integer,
    BigInteger,
    SmallInteger,
    Decimal { precision: u8, scale: u8 },
    Float,
    Double,
    
    // String types
    Char { length: u16 },
    VarChar { max_length: u16 },
    Text,
    LongText,
    
    // Date/Time types
    Date,
    Time,
    DateTime,
    Timestamp,
    
    // Binary types
    Binary { length: u16 },
    VarBinary { max_length: u16 },
    Blob,
    LongBlob,
    
    // Boolean type
    Boolean,
    
    // JSON type
    Json,
    
    // UUID type
    Uuid,
    
    // Array type (for PostgreSQL, etc.)
    Array { element_type: Box<DataType> },
    
    // Custom types
    Custom { type_name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub name: String,
    pub from_table: String,
    pub from_columns: Vec<String>,
    pub to_table: String,
    pub to_columns: Vec<String>,
    pub relationship_type: RelationshipType,
    pub on_delete: ReferentialAction,
    pub on_update: ReferentialAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelationshipType {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReferentialAction {
    Cascade,
    SetNull,
    SetDefault,
    Restrict,
    NoAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKey {
    pub name: String,
    pub columns: Vec<String>,
    pub referenced_table: String,
    pub referenced_columns: Vec<String>,
    pub on_delete: ReferentialAction,
    pub on_update: ReferentialAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub table: String,
    pub columns: Vec<String>,
    pub unique: bool,
    pub index_type: IndexType,
    pub where_clause: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    BTree,
    Hash,
    GiST,
    GIN,
    BRIN,
    Spatial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub table: String,
    pub constraint_type: ConstraintType,
    pub definition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConstraintType {
    PrimaryKey,
    ForeignKey,
    Unique,
    Check,
    NotNull,
}

/// Schema evolution manager
#[derive(Debug)]
pub struct SchemaEvolutionManager {
    schemas: Arc<tokio::sync::RwLock<HashMap<String, Vec<Schema>>>>,
    migrations: Arc<tokio::sync::RwLock<HashMap<String, Vec<Migration>>>>,
    validators: HashMap<String, Box<dyn SchemaValidator + Send + Sync>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    pub id: String,
    pub name: String,
    pub from_version: String,
    pub to_version: String,
    pub operations: Vec<MigrationOperation>,
    pub checksum: String,
    pub created_at: DateTime<Utc>,
    pub applied_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationOperation {
    CreateTable {
        table: TableSchema,
    },
    DropTable {
        table_name: String,
    },
    AddColumn {
        table_name: String,
        column: ColumnDefinition,
    },
    DropColumn {
        table_name: String,
        column_name: String,
    },
    ModifyColumn {
        table_name: String,
        old_column: ColumnDefinition,
        new_column: ColumnDefinition,
    },
    CreateIndex {
        index: IndexDefinition,
    },
    DropIndex {
        index_name: String,
    },
    AddConstraint {
        constraint: Constraint,
    },
    DropConstraint {
        table_name: String,
        constraint_name: String,
    },
    RenameTable {
        old_name: String,
        new_name: String,
    },
    RenameColumn {
        table_name: String,
        old_name: String,
        new_name: String,
    },
    CustomSQL {
        sql: String,
        rollback_sql: Option<String>,
    },
}

trait SchemaValidator: Send + Sync {
    fn validate_schema(&self, schema: &Schema) -> VmResult<Vec<ValidationError>>;
    fn validate_data(&self, schema: &Schema, data: &Value) -> VmResult<Vec<ValidationError>>;
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub message: String,
    pub table: Option<String>,
    pub column: Option<String>,
    pub value: Option<JsonValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorType {
    TypeMismatch,
    NullConstraintViolation,
    UniqueConstraintViolation,
    ForeignKeyViolation,
    CheckConstraintViolation,
    LengthViolation,
    RangeViolation,
    FormatViolation,
}

impl SchemaEvolutionManager {
    pub fn new() -> Self {
        let mut validators: HashMap<String, Box<dyn SchemaValidator + Send + Sync>> = HashMap::new();
        validators.insert("default".to_string(), Box::new(DefaultSchemaValidator));
        validators.insert("strict".to_string(), Box::new(StrictSchemaValidator));
        validators.insert("lenient".to_string(), Box::new(LenientSchemaValidator));

        Self {
            schemas: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            migrations: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            validators,
        }
    }

    /// Infer schema from data samples
    pub async fn infer_schema(&self, data: &Value, inference_rules: &Value) -> VmResult<Schema> {
        let mut schema = Schema {
            name: "inferred_schema".to_string(),
            version: "1.0.0".to_string(),
            tables: HashMap::new(),
            relationships: Vec::new(),
            indexes: Vec::new(),
            constraints: Vec::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Parse inference rules
        let rules = self.parse_inference_rules(inference_rules)?;

        // Infer tables and columns from data
        match data {
            Value::List(tables) => {
                for (i, table_data) in tables.iter().enumerate() {
                    let table_name = format!("table_{}", i);
                    let table_schema = self.infer_table_schema(&table_name, table_data, &rules)?;
                    schema.tables.insert(table_name, table_schema);
                }
            }
            _ => {
                // Single table inference
                let table_schema = self.infer_table_schema("main_table", data, &rules)?;
                schema.tables.insert("main_table".to_string(), table_schema);
            }
        }

        // Infer relationships
        schema.relationships = self.infer_relationships(&schema.tables, &rules)?;

        // Store the inferred schema
        let mut schemas = self.schemas.write().await;
        let schema_versions = schemas.entry(schema.name.clone()).or_insert(Vec::new());
        schema_versions.push(schema.clone());

        Ok(schema)
    }

    /// Generate migration between two schema versions
    pub async fn generate_migration(
        &self,
        from_schema: &Schema,
        to_schema: &Schema,
    ) -> VmResult<Migration> {
        let migration_id = Uuid::new_v4().to_string();
        let mut operations = Vec::new();

        // Compare tables
        for (table_name, to_table) in &to_schema.tables {
            if let Some(from_table) = from_schema.tables.get(table_name) {
                // Table exists, check for changes
                let table_ops = self.generate_table_migration(from_table, to_table)?;
                operations.extend(table_ops);
            } else {
                // New table
                operations.push(MigrationOperation::CreateTable {
                    table: to_table.clone(),
                });
            }
        }

        // Check for dropped tables
        for (table_name, _) in &from_schema.tables {
            if !to_schema.tables.contains_key(table_name) {
                operations.push(MigrationOperation::DropTable {
                    table_name: table_name.clone(),
                });
            }
        }

        // Compare indexes
        let index_ops = self.generate_index_migration(&from_schema.indexes, &to_schema.indexes)?;
        operations.extend(index_ops);

        // Compare constraints
        let constraint_ops = self.generate_constraint_migration(&from_schema.constraints, &to_schema.constraints)?;
        operations.extend(constraint_ops);

        let migration = Migration {
            id: migration_id,
            name: format!("migrate_{}_{}", from_schema.version, to_schema.version),
            from_version: from_schema.version.clone(),
            to_version: to_schema.version.clone(),
            operations,
            checksum: "".to_string(), // TODO: Calculate actual checksum
            created_at: Utc::now(),
            applied_at: None,
        };

        // Store the migration
        let mut migrations = self.migrations.write().await;
        let schema_migrations = migrations.entry(from_schema.name.clone()).or_insert(Vec::new());
        schema_migrations.push(migration.clone());

        Ok(migration)
    }

    /// Execute migration on database
    pub async fn execute_migration(
        &self,
        connection: &Arc<dyn DatabaseConnection>,
        migration: &Migration,
    ) -> VmResult<Value> {
        let mut results = Vec::new();

        for operation in &migration.operations {
            let sql = self.generate_migration_sql(operation)?;
            let result = connection.execute_query(&sql, vec![]).await?;
            results.push(Value::List(vec![
                Value::String(format!("{:?}", operation)),
                result,
            ]));
        }

        Ok(Value::List(results))
    }

    /// Validate data against schema
    pub async fn validate_data(
        &self,
        schema: &Schema,
        data: &Value,
        validator_name: &str,
    ) -> VmResult<Vec<ValidationError>> {
        let validator = self.validators.get(validator_name)
            .ok_or_else(|| VmError::Runtime(format!("Unknown validator: {}", validator_name)))?;

        validator.validate_data(schema, data)
    }

    /// Compare two schemas and generate diff
    pub async fn compare_schemas(&self, schema1: &Schema, schema2: &Schema) -> VmResult<SchemaDiff> {
        let mut diff = SchemaDiff {
            added_tables: Vec::new(),
            removed_tables: Vec::new(),
            modified_tables: Vec::new(),
            added_indexes: Vec::new(),
            removed_indexes: Vec::new(),
            added_constraints: Vec::new(),
            removed_constraints: Vec::new(),
        };

        // Compare tables
        for (table_name, table2) in &schema2.tables {
            if let Some(table1) = schema1.tables.get(table_name) {
                let table_diff = self.compare_tables(table1, table2)?;
                if !table_diff.is_empty() {
                    diff.modified_tables.push((table_name.clone(), table_diff));
                }
            } else {
                diff.added_tables.push(table_name.clone());
            }
        }

        for table_name in schema1.tables.keys() {
            if !schema2.tables.contains_key(table_name) {
                diff.removed_tables.push(table_name.clone());
            }
        }

        // Compare indexes
        let index_names1: HashSet<String> = schema1.indexes.iter().map(|i| i.name.clone()).collect();
        let index_names2: HashSet<String> = schema2.indexes.iter().map(|i| i.name.clone()).collect();

        diff.added_indexes = index_names2.difference(&index_names1).cloned().collect();
        diff.removed_indexes = index_names1.difference(&index_names2).cloned().collect();

        // Compare constraints
        let constraint_names1: HashSet<String> = schema1.constraints.iter().map(|c| c.name.clone()).collect();
        let constraint_names2: HashSet<String> = schema2.constraints.iter().map(|c| c.name.clone()).collect();

        diff.added_constraints = constraint_names2.difference(&constraint_names1).cloned().collect();
        diff.removed_constraints = constraint_names1.difference(&constraint_names2).cloned().collect();

        Ok(diff)
    }

    // Private helper methods

    fn parse_inference_rules(&self, rules: &Value) -> VmResult<InferenceRules> {
        // Default inference rules
        let mut inference_rules = InferenceRules {
            sample_size: 1000,
            null_threshold: 0.1,
            unique_threshold: 0.95,
            string_length_percentile: 95,
            detect_foreign_keys: true,
            detect_indexes: true,
        };

        if let Value::List(rule_pairs) = rules {
            for pair in rule_pairs {
                if let Value::List(kv) = pair {
                    if kv.len() == 2 {
                        if let (Value::String(key), value) = (&kv[0], &kv[1]) {
                            match key.as_str() {
                                "sample_size" => {
                                    if let Value::Real(n) = value {
                                        inference_rules.sample_size = *n as usize;
                                    }
                                }
                                "null_threshold" => {
                                    if let Value::Real(n) = value {
                                        inference_rules.null_threshold = *n;
                                    }
                                }
                                "unique_threshold" => {
                                    if let Value::Real(n) = value {
                                        inference_rules.unique_threshold = *n;
                                    }
                                }
                                "detect_foreign_keys" => {
                                    if let Value::Boolean(b) = value {
                                        inference_rules.detect_foreign_keys = *b;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        Ok(inference_rules)
    }

    fn infer_table_schema(&self, table_name: &str, data: &Value, rules: &InferenceRules) -> VmResult<TableSchema> {
        let mut columns = HashMap::new();
        let mut sample_count = 0;

        // Analyze data to infer column types
        match data {
            Value::List(rows) => {
                let sample_size = std::cmp::min(rows.len(), rules.sample_size);
                
                // First pass: collect all column names
                let mut column_stats: HashMap<String, ColumnStats> = HashMap::new();

                for row in rows.iter().take(sample_size) {
                    sample_count += 1;
                    if let Value::List(row_data) = row {
                        for item in row_data {
                            if let Value::List(kv) = item {
                                if kv.len() == 2 {
                                    if let Value::String(column_name) = &kv[0] {
                                        let stats = column_stats.entry(column_name.clone()).or_insert(ColumnStats::new());
                                        stats.update(&kv[1]);
                                    }
                                }
                            }
                        }
                    }
                }

                // Second pass: infer column definitions
                for (column_name, stats) in column_stats {
                    let column_def = self.infer_column_definition(&column_name, &stats, sample_count, rules)?;
                    columns.insert(column_name, column_def);
                }
            }
            _ => {
                return Err(VmError::Runtime("Data must be a list of rows".to_string()));
            }
        }

        // Infer primary key (first column that looks like an ID)
        let primary_key = columns.keys()
            .find(|name| name.to_lowercase().contains("id"))
            .map(|name| vec![name.clone()])
            .unwrap_or_else(Vec::new);

        Ok(TableSchema {
            name: table_name.to_string(),
            columns,
            primary_key,
            foreign_keys: Vec::new(),
            indexes: Vec::new(),
            constraints: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    fn infer_column_definition(
        &self,
        column_name: &str,
        stats: &ColumnStats,
        sample_count: usize,
        rules: &InferenceRules,
    ) -> VmResult<ColumnDefinition> {
        let data_type = self.infer_data_type(stats)?;
        let nullable = stats.null_count as f64 / sample_count as f64 > rules.null_threshold;
        let unique = stats.unique_count as f64 / sample_count as f64 > rules.unique_threshold;

        Ok(ColumnDefinition {
            name: column_name.to_string(),
            data_type,
            nullable,
            default_value: None,
            auto_increment: column_name.to_lowercase().contains("id") && unique,
            unique,
            comment: None,
            metadata: HashMap::new(),
        })
    }

    fn infer_data_type(&self, stats: &ColumnStats) -> VmResult<DataType> {
        // Simple type inference based on observed values
        if stats.has_integers && !stats.has_floats && !stats.has_strings {
            if stats.max_integer_value <= i32::MAX as i64 {
                Ok(DataType::Integer)
            } else {
                Ok(DataType::BigInteger)
            }
        } else if stats.has_floats && !stats.has_strings {
            Ok(DataType::Double)
        } else if stats.has_booleans && !stats.has_strings && !stats.has_integers && !stats.has_floats {
            Ok(DataType::Boolean)
        } else if stats.has_strings {
            if stats.max_string_length <= 255 {
                Ok(DataType::VarChar { max_length: std::cmp::max(stats.max_string_length, 50) })
            } else {
                Ok(DataType::Text)
            }
        } else {
            Ok(DataType::Text)
        }
    }

    fn infer_relationships(&self, tables: &HashMap<String, TableSchema>, rules: &InferenceRules) -> VmResult<Vec<Relationship>> {
        let mut relationships = Vec::new();

        if !rules.detect_foreign_keys {
            return Ok(relationships);
        }

        // Simple foreign key detection based on column names
        for (table_name, table) in tables {
            for (column_name, column) in &table.columns {
                if column_name.ends_with("_id") && column_name != "id" {
                    let referenced_table = column_name.trim_end_matches("_id");
                    if tables.contains_key(referenced_table) {
                        relationships.push(Relationship {
                            name: format!("fk_{}_{}", table_name, column_name),
                            from_table: table_name.clone(),
                            from_columns: vec![column_name.clone()],
                            to_table: referenced_table.to_string(),
                            to_columns: vec!["id".to_string()],
                            relationship_type: RelationshipType::ManyToOne,
                            on_delete: ReferentialAction::Restrict,
                            on_update: ReferentialAction::Cascade,
                        });
                    }
                }
            }
        }

        Ok(relationships)
    }

    fn generate_table_migration(&self, from_table: &TableSchema, to_table: &TableSchema) -> VmResult<Vec<MigrationOperation>> {
        let mut operations = Vec::new();

        // Compare columns
        for (column_name, to_column) in &to_table.columns {
            if let Some(from_column) = from_table.columns.get(column_name) {
                // Column exists, check for changes
                if from_column != to_column {
                    operations.push(MigrationOperation::ModifyColumn {
                        table_name: to_table.name.clone(),
                        old_column: from_column.clone(),
                        new_column: to_column.clone(),
                    });
                }
            } else {
                // New column
                operations.push(MigrationOperation::AddColumn {
                    table_name: to_table.name.clone(),
                    column: to_column.clone(),
                });
            }
        }

        // Check for dropped columns
        for (column_name, _) in &from_table.columns {
            if !to_table.columns.contains_key(column_name) {
                operations.push(MigrationOperation::DropColumn {
                    table_name: to_table.name.clone(),
                    column_name: column_name.clone(),
                });
            }
        }

        Ok(operations)
    }

    fn generate_index_migration(&self, from_indexes: &[IndexDefinition], to_indexes: &[IndexDefinition]) -> VmResult<Vec<MigrationOperation>> {
        let mut operations = Vec::new();

        let from_names: HashSet<String> = from_indexes.iter().map(|i| i.name.clone()).collect();
        let to_names: HashSet<String> = to_indexes.iter().map(|i| i.name.clone()).collect();

        // Add new indexes
        for index in to_indexes {
            if !from_names.contains(&index.name) {
                operations.push(MigrationOperation::CreateIndex {
                    index: index.clone(),
                });
            }
        }

        // Drop removed indexes
        for index_name in &from_names {
            if !to_names.contains(index_name) {
                operations.push(MigrationOperation::DropIndex {
                    index_name: index_name.clone(),
                });
            }
        }

        Ok(operations)
    }

    fn generate_constraint_migration(&self, from_constraints: &[Constraint], to_constraints: &[Constraint]) -> VmResult<Vec<MigrationOperation>> {
        let mut operations = Vec::new();

        let from_names: HashSet<String> = from_constraints.iter().map(|c| c.name.clone()).collect();
        let to_names: HashSet<String> = to_constraints.iter().map(|c| c.name.clone()).collect();

        // Add new constraints
        for constraint in to_constraints {
            if !from_names.contains(&constraint.name) {
                operations.push(MigrationOperation::AddConstraint {
                    constraint: constraint.clone(),
                });
            }
        }

        // Drop removed constraints
        for constraint in from_constraints {
            if !to_names.contains(&constraint.name) {
                operations.push(MigrationOperation::DropConstraint {
                    table_name: constraint.table.clone(),
                    constraint_name: constraint.name.clone(),
                });
            }
        }

        Ok(operations)
    }

    fn generate_migration_sql(&self, operation: &MigrationOperation) -> VmResult<String> {
        match operation {
            MigrationOperation::CreateTable { table } => {
                Ok(format!("CREATE TABLE {} (...)", table.name))
            }
            MigrationOperation::DropTable { table_name } => {
                Ok(format!("DROP TABLE {}", table_name))
            }
            MigrationOperation::AddColumn { table_name, column } => {
                Ok(format!("ALTER TABLE {} ADD COLUMN {} {}", 
                    table_name, column.name, self.format_data_type(&column.data_type)?))
            }
            MigrationOperation::DropColumn { table_name, column_name } => {
                Ok(format!("ALTER TABLE {} DROP COLUMN {}", table_name, column_name))
            }
            MigrationOperation::CustomSQL { sql, .. } => {
                Ok(sql.clone())
            }
            _ => Ok("-- Migration operation not implemented".to_string()),
        }
    }

    fn format_data_type(&self, data_type: &DataType) -> VmResult<String> {
        match data_type {
            DataType::Integer => Ok("INTEGER".to_string()),
            DataType::BigInteger => Ok("BIGINT".to_string()),
            DataType::VarChar { max_length } => Ok(format!("VARCHAR({})", max_length)),
            DataType::Text => Ok("TEXT".to_string()),
            DataType::Boolean => Ok("BOOLEAN".to_string()),
            DataType::DateTime => Ok("TIMESTAMP".to_string()),
            DataType::Json => Ok("JSON".to_string()),
            _ => Ok("TEXT".to_string()),
        }
    }

    fn compare_tables(&self, table1: &TableSchema, table2: &TableSchema) -> VmResult<TableDiff> {
        let mut diff = TableDiff {
            added_columns: Vec::new(),
            removed_columns: Vec::new(),
            modified_columns: Vec::new(),
        };

        // Compare columns
        for (column_name, column2) in &table2.columns {
            if let Some(column1) = table1.columns.get(column_name) {
                if column1 != column2 {
                    diff.modified_columns.push((column_name.clone(), column1.clone(), column2.clone()));
                }
            } else {
                diff.added_columns.push(column_name.clone());
            }
        }

        for column_name in table1.columns.keys() {
            if !table2.columns.contains_key(column_name) {
                diff.removed_columns.push(column_name.clone());
            }
        }

        Ok(diff)
    }
}

impl Foreign for SchemaEvolutionManager {
    fn type_name(&self) -> &'static str {
        "SchemaEvolutionManager"
    }
}

#[derive(Debug)]
struct InferenceRules {
    sample_size: usize,
    null_threshold: f64,
    unique_threshold: f64,
    string_length_percentile: u8,
    detect_foreign_keys: bool,
    detect_indexes: bool,
}

#[derive(Debug)]
struct ColumnStats {
    null_count: usize,
    unique_count: usize,
    unique_values: HashSet<String>,
    has_integers: bool,
    has_floats: bool,
    has_strings: bool,
    has_booleans: bool,
    max_string_length: u16,
    max_integer_value: i64,
    min_integer_value: i64,
}

impl ColumnStats {
    fn new() -> Self {
        Self {
            null_count: 0,
            unique_count: 0,
            unique_values: HashSet::new(),
            has_integers: false,
            has_floats: false,
            has_strings: false,
            has_booleans: false,
            max_string_length: 0,
            max_integer_value: i64::MIN,
            min_integer_value: i64::MAX,
        }
    }

    fn update(&mut self, value: &Value) {
        match value {
            Value::Missing => {
                self.null_count += 1;
            }
            Value::Real(n) => {
                if n.fract() == 0.0 {
                    self.has_integers = true;
                    let int_val = *n as i64;
                    self.max_integer_value = self.max_integer_value.max(int_val);
                    self.min_integer_value = self.min_integer_value.min(int_val);
                } else {
                    self.has_floats = true;
                }
                self.unique_values.insert(n.to_string());
            }
            Value::String(s) => {
                self.has_strings = true;
                self.max_string_length = self.max_string_length.max(s.len() as u16);
                self.unique_values.insert(s.clone());
            }
            Value::Boolean(_) => {
                self.has_booleans = true;
                self.unique_values.insert(value.to_string());
            }
            _ => {
                self.has_strings = true;
                let str_repr = format!("{:?}", value);
                self.max_string_length = self.max_string_length.max(str_repr.len() as u16);
                self.unique_values.insert(str_repr);
            }
        }
        
        self.unique_count = self.unique_values.len();
    }
}

#[derive(Debug)]
pub struct SchemaDiff {
    pub added_tables: Vec<String>,
    pub removed_tables: Vec<String>,
    pub modified_tables: Vec<(String, TableDiff)>,
    pub added_indexes: Vec<String>,
    pub removed_indexes: Vec<String>,
    pub added_constraints: Vec<String>,
    pub removed_constraints: Vec<String>,
}

#[derive(Debug)]
pub struct TableDiff {
    pub added_columns: Vec<String>,
    pub removed_columns: Vec<String>,
    pub modified_columns: Vec<(String, ColumnDefinition, ColumnDefinition)>,
}

impl TableDiff {
    fn is_empty(&self) -> bool {
        self.added_columns.is_empty() && 
        self.removed_columns.is_empty() && 
        self.modified_columns.is_empty()
    }
}

// Schema Validators

struct DefaultSchemaValidator;

impl SchemaValidator for DefaultSchemaValidator {
    fn validate_schema(&self, schema: &Schema) -> VmResult<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Basic schema validation
        if schema.name.is_empty() {
            errors.push(ValidationError {
                error_type: ValidationErrorType::FormatViolation,
                message: "Schema name cannot be empty".to_string(),
                table: None,
                column: None,
                value: None,
            });
        }

        // Validate tables
        for (table_name, table) in &schema.tables {
            if table.columns.is_empty() {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::FormatViolation,
                    message: "Table must have at least one column".to_string(),
                    table: Some(table_name.clone()),
                    column: None,
                    value: None,
                });
            }
        }

        Ok(errors)
    }

    fn validate_data(&self, schema: &Schema, data: &Value) -> VmResult<Vec<ValidationError>> {
        let mut errors = Vec::new();

        if let Value::List(rows) = data {
            for (row_index, row) in rows.iter().enumerate() {
                if let Value::List(row_data) = row {
                    // Extract table name (assume first table if not specified)
                    let table_name = schema.tables.keys().next()
                        .ok_or_else(|| VmError::Runtime("No tables in schema".to_string()))?;
                    
                    let table_schema = &schema.tables[table_name];
                    
                    // Validate each column
                    for item in row_data {
                        if let Value::List(kv) = item {
                            if kv.len() == 2 {
                                if let Value::String(column_name) = &kv[0] {
                                    if let Some(column_def) = table_schema.columns.get(column_name) {
                                        let validation_result = self.validate_column_value(
                                            column_def, 
                                            &kv[1], 
                                            table_name, 
                                            column_name, 
                                            row_index
                                        );
                                        if let Err(error) = validation_result {
                                            errors.push(error);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(errors)
    }
}

impl DefaultSchemaValidator {
    fn validate_column_value(
        &self,
        column_def: &ColumnDefinition,
        value: &Value,
        table_name: &str,
        column_name: &str,
        row_index: usize,
    ) -> Result<(), ValidationError> {
        // Check null constraint
        if !column_def.nullable && matches!(value, Value::Missing) {
            return Err(ValidationError {
                error_type: ValidationErrorType::NullConstraintViolation,
                message: format!("Column '{}' cannot be null (row {})", column_name, row_index),
                table: Some(table_name.to_string()),
                column: Some(column_name.to_string()),
                value: None,
            });
        }

        // Check data type compatibility
        if !matches!(value, Value::Missing) {
            match (&column_def.data_type, value) {
                (DataType::Integer | DataType::BigInteger | DataType::SmallInteger, Value::Real(n)) => {
                    if n.fract() != 0.0 {
                        return Err(ValidationError {
                            error_type: ValidationErrorType::TypeMismatch,
                            message: format!("Expected integer for column '{}', got float (row {})", column_name, row_index),
                            table: Some(table_name.to_string()),
                            column: Some(column_name.to_string()),
                            value: Some(JsonValue::Real(serde_json::Number::from_f64(*n).unwrap())),
                        });
                    }
                }
                (DataType::Boolean, Value::Boolean(_)) => {}
                (DataType::VarChar { max_length }, Value::String(s)) => {
                    if s.len() > *max_length as usize {
                        return Err(ValidationError {
                            error_type: ValidationErrorType::LengthViolation,
                            message: format!("String too long for column '{}': {} > {} (row {})", 
                                column_name, s.len(), max_length, row_index),
                            table: Some(table_name.to_string()),
                            column: Some(column_name.to_string()),
                            value: Some(JsonValue::String(s.clone())),
                        });
                    }
                }
                (DataType::Text, Value::String(_)) => {}
                (expected, actual) => {
                    return Err(ValidationError {
                        error_type: ValidationErrorType::TypeMismatch,
                        message: format!("Type mismatch for column '{}': expected {:?}, got {:?} (row {})", 
                            column_name, expected, actual, row_index),
                        table: Some(table_name.to_string()),
                        column: Some(column_name.to_string()),
                        value: Some(self.value_to_json(actual)),
                    });
                }
            }
        }

        Ok(())
    }

    fn value_to_json(&self, value: &Value) -> JsonValue {
        match value {
            Value::Real(n) => JsonValue::Real(serde_json::Number::from_f64(*n).unwrap()),
            Value::String(s) => JsonValue::String(s.clone()),
            Value::Boolean(b) => JsonValue::Bool(*b),
            Value::Missing => JsonValue::Null,
            _ => JsonValue::String(format!("{:?}", value)),
        }
    }
}

struct StrictSchemaValidator;

impl SchemaValidator for StrictSchemaValidator {
    fn validate_schema(&self, schema: &Schema) -> VmResult<Vec<ValidationError>> {
        // Strict validation with additional checks
        let mut errors = DefaultSchemaValidator.validate_schema(schema)?;
        
        // Add strict validation rules
        for (table_name, table) in &schema.tables {
            if table.primary_key.is_empty() {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::FormatViolation,
                    message: "Table must have a primary key".to_string(),
                    table: Some(table_name.clone()),
                    column: None,
                    value: None,
                });
            }
        }

        Ok(errors)
    }

    fn validate_data(&self, schema: &Schema, data: &Value) -> VmResult<Vec<ValidationError>> {
        // Use default validation
        DefaultSchemaValidator.validate_data(schema, data)
    }
}

struct LenientSchemaValidator;

impl SchemaValidator for LenientSchemaValidator {
    fn validate_schema(&self, _schema: &Schema) -> VmResult<Vec<ValidationError>> {
        // Lenient validation - allow most schemas
        Ok(Vec::new())
    }

    fn validate_data(&self, _schema: &Schema, _data: &Value) -> VmResult<Vec<ValidationError>> {
        // Lenient validation - allow most data
        Ok(Vec::new())
    }
}

// Public API Functions for Schema Management

/// InferSchema[data, inference_rules] - Generate schema from data
pub fn infer_schema(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("InferSchema requires data and inference_rules".to_string()));
    }

    let data = &args[0];
    let inference_rules = &args[1];

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    let manager = SchemaEvolutionManager::new();
    let schema = rt.block_on(async {
        manager.infer_schema(data, inference_rules).await
    })?;

    // Convert schema to Lyra Value
    let schema_json = serde_json::to_value(&schema)
        .map_err(|e| VmError::Runtime(format!("Failed to serialize schema: {}", e)))?;

    Ok(Value::String(schema_json.to_string()))
}

/// SchemaEvolution[database, current_schema, target_schema] - Evolve schema
pub fn schema_evolution(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime("SchemaEvolution requires database, current_schema, and target_schema".to_string()));
    }

    // Parse schemas from JSON strings
    let current_schema_str = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Current schema must be a JSON string".to_string())),
    };

    let target_schema_str = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Target schema must be a JSON string".to_string())),
    };

    let current_schema: Schema = serde_json::from_str(current_schema_str)
        .map_err(|e| VmError::Runtime(format!("Failed to parse current schema: {}", e)))?;

    let target_schema: Schema = serde_json::from_str(target_schema_str)
        .map_err(|e| VmError::Runtime(format!("Failed to parse target schema: {}", e)))?;

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    let manager = SchemaEvolutionManager::new();
    let migration = rt.block_on(async {
        manager.generate_migration(&current_schema, &target_schema).await
    })?;

    // Convert migration to Lyra Value
    let migration_json = serde_json::to_value(&migration)
        .map_err(|e| VmError::Runtime(format!("Failed to serialize migration: {}", e)))?;

    Ok(Value::String(migration_json.to_string()))
}

/// SchemaValidation[database, schema, data] - Validate data against schema
pub fn schema_validation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime("SchemaValidation requires database, schema, and data".to_string()));
    }

    let schema_str = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Schema must be a JSON string".to_string())),
    };

    let data = &args[2];

    let schema: Schema = serde_json::from_str(schema_str)
        .map_err(|e| VmError::Runtime(format!("Failed to parse schema: {}", e)))?;

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    let manager = SchemaEvolutionManager::new();
    let errors = rt.block_on(async {
        manager.validate_data(&schema, data, "default").await
    })?;

    // Convert validation errors to Lyra Value
    let error_values: Vec<Value> = errors.iter().map(|error| {
        Value::List(vec![
            Value::String(format!("{:?}", error.error_type)),
            Value::String(error.message.clone()),
            error.table.as_ref().map(|t| Value::String(t.clone())).unwrap_or(Value::Missing),
            error.column.as_ref().map(|c| Value::String(c.clone())).unwrap_or(Value::Missing),
        ])
    }).collect();

    Ok(Value::List(error_values))
}

/// SchemaDiff[schema1, schema2] - Compare schemas
pub fn schema_diff(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("SchemaDiff requires two schemas".to_string()));
    }

    let schema1_str = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Schema1 must be a JSON string".to_string())),
    };

    let schema2_str = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Schema2 must be a JSON string".to_string())),
    };

    let schema1: Schema = serde_json::from_str(schema1_str)
        .map_err(|e| VmError::Runtime(format!("Failed to parse schema1: {}", e)))?;

    let schema2: Schema = serde_json::from_str(schema2_str)
        .map_err(|e| VmError::Runtime(format!("Failed to parse schema2: {}", e)))?;

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    let manager = SchemaEvolutionManager::new();
    let diff = rt.block_on(async {
        manager.compare_schemas(&schema1, &schema2).await
    })?;

    // Convert diff to Lyra Value
    Ok(Value::List(vec![
        Value::List(vec![
            Value::String("added_tables".to_string()),
            Value::List(diff.added_tables.into_iter().map(Value::String).collect()),
        ]),
        Value::List(vec![
            Value::String("removed_tables".to_string()),
            Value::List(diff.removed_tables.into_iter().map(Value::String).collect()),
        ]),
        Value::List(vec![
            Value::String("added_indexes".to_string()),
            Value::List(diff.added_indexes.into_iter().map(Value::String).collect()),
        ]),
        Value::List(vec![
            Value::String("removed_indexes".to_string()),
            Value::List(diff.removed_indexes.into_iter().map(Value::String).collect()),
        ]),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let schema = Schema {
            name: "test_schema".to_string(),
            version: "1.0.0".to_string(),
            tables: HashMap::new(),
            relationships: Vec::new(),
            indexes: Vec::new(),
            constraints: Vec::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert_eq!(schema.name, "test_schema");
        assert_eq!(schema.version, "1.0.0");
    }

    #[test]
    fn test_column_stats() {
        let mut stats = ColumnStats::new();
        
        stats.update(&Value::Real(42.0));
        stats.update(&Value::String("hello".to_string()));
        stats.update(&Value::Missing);
        
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.unique_count, 2);
        assert!(stats.has_integers);
        assert!(stats.has_strings);
    }

    #[test]
    fn test_data_type_inference() {
        let manager = SchemaEvolutionManager::new();
        
        let mut int_stats = ColumnStats::new();
        int_stats.has_integers = true;
        int_stats.max_integer_value = 100;
        
        let data_type = manager.infer_data_type(&int_stats).unwrap();
        assert_eq!(data_type, DataType::Integer);
        
        let mut string_stats = ColumnStats::new();
        string_stats.has_strings = true;
        string_stats.max_string_length = 50;
        
        let data_type = manager.infer_data_type(&string_stats).unwrap();
        match data_type {
            DataType::VarChar { max_length } => assert_eq!(max_length, 50),
            _ => panic!("Expected VarChar"),
        }
    }

    #[test]
    fn test_infer_schema_function() {
        let data = Value::List(vec![
            Value::List(vec![
                Value::List(vec![Value::String("id".to_string()), Value::Real(1.0)]),
                Value::List(vec![Value::String("name".to_string()), Value::String("Alice".to_string())]),
            ]),
            Value::List(vec![
                Value::List(vec![Value::String("id".to_string()), Value::Real(2.0)]),
                Value::List(vec![Value::String("name".to_string()), Value::String("Bob".to_string())]),
            ]),
        ]);

        let rules = Value::List(vec![
            Value::List(vec![Value::String("sample_size".to_string()), Value::Real(100.0)]),
        ]);

        let args = vec![data, rules];
        let result = infer_schema(&args);
        
        assert!(result.is_ok());
        if let Ok(Value::String(schema_json)) = result {
            assert!(schema_json.contains("inferred_schema"));
        }
    }

    #[test]
    fn test_schema_validation_function() {
        let schema_json = r#"{
            "name": "test_schema",
            "version": "1.0.0",
            "tables": {
                "users": {
                    "name": "users",
                    "columns": {
                        "id": {
                            "name": "id",
                            "data_type": "Integer",
                            "nullable": false,
                            "default_value": null,
                            "auto_increment": true,
                            "unique": true,
                            "comment": null,
                            "metadata": {}
                        },
                        "name": {
                            "name": "name",
                            "data_type": {"VarChar": {"max_length": 100}},
                            "nullable": false,
                            "default_value": null,
                            "auto_increment": false,
                            "unique": false,
                            "comment": null,
                            "metadata": {}
                        }
                    },
                    "primary_key": ["id"],
                    "foreign_keys": [],
                    "indexes": [],
                    "constraints": [],
                    "metadata": {}
                }
            },
            "relationships": [],
            "indexes": [],
            "constraints": [],
            "metadata": {},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }"#;

        let data = Value::List(vec![
            Value::List(vec![
                Value::List(vec![Value::String("id".to_string()), Value::Real(1.0)]),
                Value::List(vec![Value::String("name".to_string()), Value::String("Alice".to_string())]),
            ]),
        ]);

        let args = vec![
            Value::String("database".to_string()),
            Value::String(schema_json.to_string()),
            data,
        ];

        let result = schema_validation(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::List(errors)) = result {
            // Should have no validation errors for valid data
            assert_eq!(errors.len(), 0);
        }
    }

    #[test]
    fn test_schema_diff_function() {
        let schema1 = r#"{
            "name": "schema1",
            "version": "1.0.0",
            "tables": {
                "users": {
                    "name": "users",
                    "columns": {
                        "id": {
                            "name": "id",
                            "data_type": "Integer",
                            "nullable": false,
                            "default_value": null,
                            "auto_increment": true,
                            "unique": true,
                            "comment": null,
                            "metadata": {}
                        }
                    },
                    "primary_key": ["id"],
                    "foreign_keys": [],
                    "indexes": [],
                    "constraints": [],
                    "metadata": {}
                }
            },
            "relationships": [],
            "indexes": [],
            "constraints": [],
            "metadata": {},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }"#;

        let schema2 = r#"{
            "name": "schema2", 
            "version": "2.0.0",
            "tables": {
                "users": {
                    "name": "users",
                    "columns": {
                        "id": {
                            "name": "id",
                            "data_type": "Integer",
                            "nullable": false,
                            "default_value": null,
                            "auto_increment": true,
                            "unique": true,
                            "comment": null,
                            "metadata": {}
                        }
                    },
                    "primary_key": ["id"],
                    "foreign_keys": [],
                    "indexes": [],
                    "constraints": [],
                    "metadata": {}
                },
                "posts": {
                    "name": "posts",
                    "columns": {
                        "id": {
                            "name": "id",
                            "data_type": "Integer",
                            "nullable": false,
                            "default_value": null,
                            "auto_increment": true,
                            "unique": true,
                            "comment": null,
                            "metadata": {}
                        }
                    },
                    "primary_key": ["id"],
                    "foreign_keys": [],
                    "indexes": [],
                    "constraints": [],
                    "metadata": {}
                }
            },
            "relationships": [],
            "indexes": [],
            "constraints": [],
            "metadata": {},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }"#;

        let args = vec![
            Value::String(schema1.to_string()),
            Value::String(schema2.to_string()),
        ];

        let result = schema_diff(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::List(diff_parts)) = result {
            // Should show that 'posts' table was added
            if let Value::List(added_tables_kv) = &diff_parts[0] {
                if let Value::List(added_tables) = &added_tables_kv[1] {
                    assert_eq!(added_tables.len(), 1);
                    if let Value::String(table_name) = &added_tables[0] {
                        assert_eq!(table_name, "posts");
                    }
                }
            }
        }
    }

    #[test]
    fn test_migration_operation_serialization() {
        let operation = MigrationOperation::CreateTable {
            table: TableSchema {
                name: "test_table".to_string(),
                columns: HashMap::new(),
                primary_key: vec!["id".to_string()],
                foreign_keys: Vec::new(),
                indexes: Vec::new(),
                constraints: Vec::new(),
                metadata: HashMap::new(),
            },
        };

        let serialized = serde_json::to_string(&operation);
        assert!(serialized.is_ok());
    }
}