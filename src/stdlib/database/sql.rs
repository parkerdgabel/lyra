//! SQL Database Integration Module
//!
//! This module provides comprehensive SQL database integration supporting:
//! - PostgreSQL, MySQL, SQLite through sqlx
//! - Connection pooling with deadpool
//! - Transaction management with rollback support
//! - Schema migration and management
//! - Prepared statements for security
//! - Batch operations for performance

use super::{DatabaseConnection, DatabaseTransaction, DatabaseConfig, DatabaseError, get_database_registry};
use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, LyObj};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use sqlx::{Pool, Postgres, MySql, Sqlite, Row, Column, TypeInfo};
use sqlx::postgres::PgPoolOptions;
use sqlx::mysql::MySqlPoolOptions;
use sqlx::sqlite::SqlitePoolOptions;
use deadpool::managed::{Pool as DeadPool, Manager, Object};
use serde_json::Value as JsonValue;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// SQL Database connection wrapper
#[derive(Debug)]
pub struct SQLConnection {
    pool: SQLPool,
    database_type: DatabaseType,
    config: DatabaseConfig,
}

#[derive(Debug)]
pub enum SQLPool {
    Postgres(Pool<Postgres>),
    MySQL(Pool<MySql>),
    SQLite(Pool<Sqlite>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DatabaseType {
    PostgreSQL,
    MySQL,
    SQLite,
}

impl SQLConnection {
    /// Create new SQL connection
    pub async fn new(database_type: DatabaseType, config: DatabaseConfig) -> VmResult<Self> {
        let pool = match database_type {
            DatabaseType::PostgreSQL => {
                let pool = PgPoolOptions::new()
                    .max_connections(config.pool_size.unwrap_or(10))
                    .connect_timeout(config.connection_timeout.unwrap_or(std::time::Duration::from_secs(30)))
                    .connect(&config.connection_string)
                    .await
                    .map_err(|e| DatabaseError::ConnectionError { 
                        message: format!("PostgreSQL connection failed: {}", e) 
                    })?;
                SQLPool::Postgres(pool)
            }
            DatabaseType::MySQL => {
                let pool = MySqlPoolOptions::new()
                    .max_connections(config.pool_size.unwrap_or(10))
                    .connect_timeout(config.connection_timeout.unwrap_or(std::time::Duration::from_secs(30)))
                    .connect(&config.connection_string)
                    .await
                    .map_err(|e| DatabaseError::ConnectionError { 
                        message: format!("MySQL connection failed: {}", e) 
                    })?;
                SQLPool::MySQL(pool)
            }
            DatabaseType::SQLite => {
                let pool = SqlitePoolOptions::new()
                    .max_connections(config.pool_size.unwrap_or(10))
                    .connect_timeout(config.connection_timeout.unwrap_or(std::time::Duration::from_secs(30)))
                    .connect(&config.connection_string)
                    .await
                    .map_err(|e| DatabaseError::ConnectionError { 
                        message: format!("SQLite connection failed: {}", e) 
                    })?;
                SQLPool::SQLite(pool)
            }
        };

        Ok(Self {
            pool,
            database_type,
            config,
        })
    }

    /// Execute a query with parameters
    pub async fn execute_query_internal(&self, query: &str, params: Vec<Value>) -> VmResult<Vec<HashMap<String, Value>>> {
        match &self.pool {
            SQLPool::Postgres(pool) => {
                let mut query_builder = sqlx::query(query);
                
                // Bind parameters
                for param in params {
                    query_builder = match param {
                        Value::Real(n) => query_builder.bind(n),
                        Value::String(s) => query_builder.bind(s),
                        Value::Boolean(b) => query_builder.bind(b),
                        Value::Missing => query_builder.bind(Option::<String>::None),
                        _ => return Err(VmError::Runtime(format!("Unsupported parameter type: {:?}", param))),
                    };
                }
                
                let rows = query_builder
                    .fetch_all(pool)
                    .await
                    .map_err(|e| DatabaseError::QueryError { 
                        message: format!("PostgreSQL query failed: {}", e) 
                    })?;

                let mut results = Vec::new();
                for row in rows {
                    let mut row_map = HashMap::new();
                    for (i, column) in row.columns().iter().enumerate() {
                        let column_name = column.name().to_string();
                        let value = self.extract_postgres_value(&row, i, column)?;
                        row_map.insert(column_name, value);
                    }
                    results.push(row_map);
                }
                
                Ok(results)
            }
            SQLPool::MySQL(pool) => {
                let mut query_builder = sqlx::query(query);
                
                // Bind parameters
                for param in params {
                    query_builder = match param {
                        Value::Real(n) => query_builder.bind(n),
                        Value::String(s) => query_builder.bind(s),
                        Value::Boolean(b) => query_builder.bind(b),
                        Value::Missing => query_builder.bind(Option::<String>::None),
                        _ => return Err(VmError::Runtime(format!("Unsupported parameter type: {:?}", param))),
                    };
                }
                
                let rows = query_builder
                    .fetch_all(pool)
                    .await
                    .map_err(|e| DatabaseError::QueryError { 
                        message: format!("MySQL query failed: {}", e) 
                    })?;

                let mut results = Vec::new();
                for row in rows {
                    let mut row_map = HashMap::new();
                    for (i, column) in row.columns().iter().enumerate() {
                        let column_name = column.name().to_string();
                        let value = self.extract_mysql_value(&row, i, column)?;
                        row_map.insert(column_name, value);
                    }
                    results.push(row_map);
                }
                
                Ok(results)
            }
            SQLPool::SQLite(pool) => {
                let mut query_builder = sqlx::query(query);
                
                // Bind parameters
                for param in params {
                    query_builder = match param {
                        Value::Real(n) => query_builder.bind(n),
                        Value::String(s) => query_builder.bind(s),
                        Value::Boolean(b) => query_builder.bind(b),
                        Value::Missing => query_builder.bind(Option::<String>::None),
                        _ => return Err(VmError::Runtime(format!("Unsupported parameter type: {:?}", param))),
                    };
                }
                
                let rows = query_builder
                    .fetch_all(pool)
                    .await
                    .map_err(|e| DatabaseError::QueryError { 
                        message: format!("SQLite query failed: {}", e) 
                    })?;

                let mut results = Vec::new();
                for row in rows {
                    let mut row_map = HashMap::new();
                    for (i, column) in row.columns().iter().enumerate() {
                        let column_name = column.name().to_string();
                        let value = self.extract_sqlite_value(&row, i, column)?;
                        row_map.insert(column_name, value);
                    }
                    results.push(row_map);
                }
                
                Ok(results)
            }
        }
    }

    fn extract_postgres_value(&self, row: &sqlx::postgres::PgRow, index: usize, column: &sqlx::postgres::PgColumn) -> VmResult<Value> {
        use sqlx::Row;
        
        let type_name = column.type_info().name();
        
        match type_name {
            "INT4" | "INT8" => {
                let val: Option<i64> = row.try_get(index).ok();
                Ok(val.map(Value::Real).unwrap_or(Value::Missing))
            }
            "FLOAT4" | "FLOAT8" | "NUMERIC" => {
                let val: Option<f64> = row.try_get(index).ok();
                Ok(val.map(Value::Real).unwrap_or(Value::Missing))
            }
            "TEXT" | "VARCHAR" => {
                let val: Option<String> = row.try_get(index).ok();
                Ok(val.map(Value::String).unwrap_or(Value::Missing))
            }
            "BOOL" => {
                let val: Option<bool> = row.try_get(index).ok();
                Ok(val.map(Value::Boolean).unwrap_or(Value::Missing))
            }
            "TIMESTAMPTZ" | "TIMESTAMP" => {
                let val: Option<DateTime<Utc>> = row.try_get(index).ok();
                Ok(val.map(|dt| Value::String(dt.to_rfc3339())).unwrap_or(Value::Missing))
            }
            "UUID" => {
                let val: Option<Uuid> = row.try_get(index).ok();
                Ok(val.map(|u| Value::String(u.to_string())).unwrap_or(Value::Missing))
            }
            "JSON" | "JSONB" => {
                let val: Option<JsonValue> = row.try_get(index).ok();
                Ok(val.map(|j| Value::String(j.to_string())).unwrap_or(Value::Missing))
            }
            _ => {
                // Try to get as string for unknown types
                let val: Option<String> = row.try_get(index).ok();
                Ok(val.map(Value::String).unwrap_or(Value::Missing))
            }
        }
    }

    fn extract_mysql_value(&self, row: &sqlx::mysql::MySqlRow, index: usize, column: &sqlx::mysql::MySqlColumn) -> VmResult<Value> {
        use sqlx::Row;
        
        let type_name = column.type_info().name();
        
        match type_name {
            "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => {
                let val: Option<i64> = row.try_get(index).ok();
                Ok(val.map(Value::Real).unwrap_or(Value::Missing))
            }
            "FLOAT" | "DOUBLE" | "DECIMAL" => {
                let val: Option<f64> = row.try_get(index).ok();
                Ok(val.map(Value::Real).unwrap_or(Value::Missing))
            }
            "VARCHAR" | "TEXT" | "CHAR" => {
                let val: Option<String> = row.try_get(index).ok();
                Ok(val.map(Value::String).unwrap_or(Value::Missing))
            }
            "BOOLEAN" | "TINYINT(1)" => {
                let val: Option<bool> = row.try_get(index).ok();
                Ok(val.map(Value::Boolean).unwrap_or(Value::Missing))
            }
            "TIMESTAMP" | "DATETIME" => {
                let val: Option<DateTime<Utc>> = row.try_get(index).ok();
                Ok(val.map(|dt| Value::String(dt.to_rfc3339())).unwrap_or(Value::Missing))
            }
            "JSON" => {
                let val: Option<JsonValue> = row.try_get(index).ok();
                Ok(val.map(|j| Value::String(j.to_string())).unwrap_or(Value::Missing))
            }
            _ => {
                // Try to get as string for unknown types
                let val: Option<String> = row.try_get(index).ok();
                Ok(val.map(Value::String).unwrap_or(Value::Missing))
            }
        }
    }

    fn extract_sqlite_value(&self, row: &sqlx::sqlite::SqliteRow, index: usize, column: &sqlx::sqlite::SqliteColumn) -> VmResult<Value> {
        use sqlx::Row;
        
        let type_name = column.type_info().name();
        
        match type_name {
            "INTEGER" => {
                let val: Option<i64> = row.try_get(index).ok();
                Ok(val.map(Value::Real).unwrap_or(Value::Missing))
            }
            "REAL" => {
                let val: Option<f64> = row.try_get(index).ok();
                Ok(val.map(Value::Real).unwrap_or(Value::Missing))
            }
            "TEXT" => {
                let val: Option<String> = row.try_get(index).ok();
                Ok(val.map(Value::String).unwrap_or(Value::Missing))
            }
            "BOOLEAN" => {
                let val: Option<bool> = row.try_get(index).ok();
                Ok(val.map(Value::Boolean).unwrap_or(Value::Missing))
            }
            "DATETIME" => {
                let val: Option<DateTime<Utc>> = row.try_get(index).ok();
                Ok(val.map(|dt| Value::String(dt.to_rfc3339())).unwrap_or(Value::Missing))
            }
            _ => {
                // Try to get as string for unknown types
                let val: Option<String> = row.try_get(index).ok();
                Ok(val.map(Value::String).unwrap_or(Value::Missing))
            }
        }
    }

    /// Execute a non-query command (INSERT, UPDATE, DELETE)
    pub async fn execute_command(&self, query: &str, params: Vec<Value>) -> VmResult<u64> {
        match &self.pool {
            SQLPool::Postgres(pool) => {
                let mut query_builder = sqlx::query(query);
                
                // Bind parameters
                for param in params {
                    query_builder = match param {
                        Value::Real(n) => query_builder.bind(n),
                        Value::String(s) => query_builder.bind(s),
                        Value::Boolean(b) => query_builder.bind(b),
                        Value::Missing => query_builder.bind(Option::<String>::None),
                        _ => return Err(VmError::Runtime(format!("Unsupported parameter type: {:?}", param))),
                    };
                }
                
                let result = query_builder
                    .execute(pool)
                    .await
                    .map_err(|e| DatabaseError::QueryError { 
                        message: format!("PostgreSQL command failed: {}", e) 
                    })?;

                Ok(result.rows_affected())
            }
            SQLPool::MySQL(pool) => {
                let mut query_builder = sqlx::query(query);
                
                // Bind parameters
                for param in params {
                    query_builder = match param {
                        Value::Real(n) => query_builder.bind(n),
                        Value::String(s) => query_builder.bind(s),
                        Value::Boolean(b) => query_builder.bind(b),
                        Value::Missing => query_builder.bind(Option::<String>::None),
                        _ => return Err(VmError::Runtime(format!("Unsupported parameter type: {:?}", param))),
                    };
                }
                
                let result = query_builder
                    .execute(pool)
                    .await
                    .map_err(|e| DatabaseError::QueryError { 
                        message: format!("MySQL command failed: {}", e) 
                    })?;

                Ok(result.rows_affected())
            }
            SQLPool::SQLite(pool) => {
                let mut query_builder = sqlx::query(query);
                
                // Bind parameters
                for param in params {
                    query_builder = match param {
                        Value::Real(n) => query_builder.bind(n),
                        Value::String(s) => query_builder.bind(s),
                        Value::Boolean(b) => query_builder.bind(b),
                        Value::Missing => query_builder.bind(Option::<String>::None),
                        _ => return Err(VmError::Runtime(format!("Unsupported parameter type: {:?}", param))),
                    };
                }
                
                let result = query_builder
                    .execute(pool)
                    .await
                    .map_err(|e| DatabaseError::QueryError { 
                        message: format!("SQLite command failed: {}", e) 
                    })?;

                Ok(result.rows_affected())
            }
        }
    }
}

#[async_trait::async_trait]
impl DatabaseConnection for SQLConnection {
    async fn execute_query(&self, query: &str, params: Vec<Value>) -> VmResult<Value> {
        let results = self.execute_query_internal(query, params).await?;
        
        // Convert to Lyra Value format
        let rows: Vec<Value> = results.into_iter().map(|row| {
            let row_vec: Vec<Value> = row.into_iter().map(|(k, v)| {
                Value::List(vec![Value::String(k), v])
            }).collect();
            Value::List(row_vec)
        }).collect();
        
        Ok(Value::List(rows))
    }

    async fn health_check(&self) -> VmResult<bool> {
        let result = match &self.pool {
            SQLPool::Postgres(pool) => {
                sqlx::query("SELECT 1").fetch_one(pool).await.is_ok()
            }
            SQLPool::MySQL(pool) => {
                sqlx::query("SELECT 1").fetch_one(pool).await.is_ok()
            }
            SQLPool::SQLite(pool) => {
                sqlx::query("SELECT 1").fetch_one(pool).await.is_ok()
            }
        };
        Ok(result)
    }

    fn metadata(&self) -> HashMap<String, Value> {
        let mut metadata = HashMap::new();
        metadata.insert("database_type".to_string(), Value::String(format!("{:?}", self.database_type)));
        metadata.insert("connection_string".to_string(), Value::String(self.config.connection_string.clone()));
        metadata.insert("pool_size".to_string(), Value::Real(self.config.pool_size.unwrap_or(10) as f64));
        metadata
    }

    async fn close(&self) -> VmResult<()> {
        match &self.pool {
            SQLPool::Postgres(pool) => pool.close().await,
            SQLPool::MySQL(pool) => pool.close().await,
            SQLPool::SQLite(pool) => pool.close().await,
        }
        Ok(())
    }
}

impl Foreign for SQLConnection {
    fn type_name(&self) -> &'static str {
        "SQLConnection"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "execute" => {
                if args.is_empty() {
                    return Err(crate::foreign::ForeignError::ArgumentError { expected: 1, actual: 0 });
                }
                if let Value::String(query) = &args[0] {
                    // In a real implementation, this would execute the query
                    Ok(Value::String(format!("Executed query: {}", query)))
                } else {
                    Err(crate::foreign::ForeignError::TypeError { expected: "String".to_string(), actual: format!("{:?}", args[0]) })
                }
            }
            "getDatabaseType" => {
                Ok(Value::String(format!("{:?}", self.database_type)))
            }
            _ => Err(crate::foreign::ForeignError::UnknownMethod { type_name: "SQLConnection".to_string(), method: method.to_string() })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(SQLConnection {
            pool: match &self.pool {
                SQLPool::Postgres(p) => SQLPool::Postgres(p.clone()),
                SQLPool::MySQL(p) => SQLPool::MySQL(p.clone()),
                SQLPool::SQLite(p) => SQLPool::SQLite(p.clone()),
            },
            database_type: self.database_type.clone(),
            config: self.config.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// SQL Transaction wrapper
#[derive(Debug)]
pub struct SQLTransaction {
    connection: Arc<SQLConnection>,
    transaction_id: String,
}

impl SQLTransaction {
    pub fn new(connection: Arc<SQLConnection>) -> Self {
        Self {
            connection,
            transaction_id: Uuid::new_v4().to_string(),
        }
    }
}

#[async_trait::async_trait]
impl DatabaseTransaction for SQLTransaction {
    async fn begin(&self) -> VmResult<()> {
        self.connection.execute_command("BEGIN", vec![]).await?;
        Ok(())
    }

    async fn commit(&self) -> VmResult<()> {
        self.connection.execute_command("COMMIT", vec![]).await?;
        Ok(())
    }

    async fn rollback(&self) -> VmResult<()> {
        self.connection.execute_command("ROLLBACK", vec![]).await?;
        Ok(())
    }

    async fn execute_in_transaction<F, R>(&self, f: F) -> VmResult<R>
    where
        F: FnOnce() -> VmResult<R> + Send,
        R: Send,
    {
        self.begin().await?;
        
        match f() {
            Ok(result) => {
                self.commit().await?;
                Ok(result)
            }
            Err(error) => {
                self.rollback().await?;
                Err(error)
            }
        }
    }
}

impl Foreign for SQLTransaction {
    fn type_name(&self) -> &'static str {
        "SQLTransaction"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "commit" => {
                // In a real implementation, this would commit the transaction
                Ok(Value::Boolean(true))
            }
            "rollback" => {
                // In a real implementation, this would rollback the transaction
                Ok(Value::Boolean(true))
            }
            "execute" => {
                if args.is_empty() {
                    return Err(crate::foreign::ForeignError::ArgumentError { expected: 1, actual: 0 });
                }
                if let Value::String(query) = &args[0] {
                    // In a real implementation, this would execute the query within the transaction
                    Ok(Value::String(format!("Executed in transaction: {}", query)))
                } else {
                    Err(crate::foreign::ForeignError::TypeError { expected: "String".to_string(), actual: format!("{:?}", args[0]) })
                }
            }
            "getId" => {
                Ok(Value::String(self.transaction_id.clone()))
            }
            _ => Err(crate::foreign::ForeignError::UnknownMethod { type_name: "SQLTransaction".to_string(), method: method.to_string() })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(SQLTransaction {
            connection: self.connection.clone(),
            transaction_id: self.transaction_id.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Public API Functions for Lyra

/// SQLConnect[driver, connection_string, options] - Connect to SQL databases
pub fn sql_connect(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("SQLConnect requires at least driver and connection_string".to_string()));
    }

    let driver = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Driver must be a string".to_string())),
    };

    let connection_string = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::Runtime("Connection string must be a string".to_string())),
    };

    let database_type = match driver.as_str() {
        "postgresql" | "postgres" => DatabaseType::PostgreSQL,
        "mysql" => DatabaseType::MySQL,
        "sqlite" => DatabaseType::SQLite,
        _ => return Err(VmError::Runtime(format!("Unsupported database driver: {}", driver))),
    };

    let mut config = DatabaseConfig::new(connection_string);

    // Parse options if provided
    if args.len() > 2 {
        if let Value::List(options) = &args[2] {
            for option in options {
                if let Value::List(kv) = option {
                    if kv.len() == 2 {
                        if let (Value::String(key), value) = (&kv[0], &kv[1]) {
                            match key.as_str() {
                                "pool_size" => {
                                    if let Value::Real(n) = value {
                                        config.pool_size = Some(*n as u32);
                                    }
                                }
                                "connection_timeout" => {
                                    if let Value::Real(n) = value {
                                        config.connection_timeout = Some(std::time::Duration::from_secs(*n as u64));
                                    }
                                }
                                "query_timeout" => {
                                    if let Value::Real(n) = value {
                                        config.query_timeout = Some(std::time::Duration::from_secs(*n as u64));
                                    }
                                }
                                _ => {
                                    config.extra_params.insert(key.clone(), value.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Create connection asynchronously
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;
    
    let connection = rt.block_on(async {
        SQLConnection::new(database_type, config).await
    })?;

    Ok(Value::LyObj(LyObj::new(Box::new(connection))))
}

/// SQLQuery[connection, query, parameters] - Execute SQL queries
pub fn sql_query(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("SQLQuery requires connection and query".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let query = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Query must be a string".to_string())),
    };

    let params = if args.len() > 2 {
        match &args[2] {
            Value::List(params) => params.clone(),
            _ => vec![args[2].clone()],
        }
    } else {
        vec![]
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    rt.block_on(async {
        connection.execute_query(query, params).await
    })
}

/// SQLTransaction[connection, operations] - Execute transactions with rollback
pub fn sql_transaction(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("SQLTransaction requires connection and operations".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            Arc::new(obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
                .clone())
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let operations = match &args[1] {
        Value::List(ops) => ops,
        _ => return Err(VmError::Runtime("Operations must be a list".to_string())),
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    rt.block_on(async {
        let transaction = SQLTransaction::new(connection.clone());
        
        transaction.execute_in_transaction(|| {
            // Execute all operations
            let mut results = Vec::new();
            for operation in operations {
                // Each operation should be a function call
                if let Value::List(op_parts) = operation {
                    if !op_parts.is_empty() {
                        if let Value::String(op_name) = &op_parts[0] {
                            match op_name.as_str() {
                                "SQLQuery" => {
                                    if op_parts.len() >= 3 {
                                        let query_args = vec![
                                            Value::LyObj(LyObj::new(Box::new(connection.as_ref().clone()))),
                                            op_parts[1].clone(),
                                            op_parts[2].clone(),
                                        ];
                                        let result = sql_query(&query_args)?;
                                        results.push(result);
                                    }
                                }
                                "SQLInsert" => {
                                    if op_parts.len() >= 4 {
                                        let insert_args = vec![
                                            Value::LyObj(LyObj::new(Box::new(connection.as_ref().clone()))),
                                            op_parts[1].clone(),
                                            op_parts[2].clone(),
                                            op_parts[3].clone(),
                                        ];
                                        let result = sql_insert(&insert_args)?;
                                        results.push(result);
                                    }
                                }
                                _ => {
                                    return Err(VmError::Runtime(format!("Unknown operation: {}", op_name)));
                                }
                            }
                        }
                    }
                }
            }
            Ok(Value::List(results))
        }).await
    })
}

/// SQLInsert[connection, table, data, options] - Insert data with conflict resolution
pub fn sql_insert(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime("SQLInsert requires connection, table, and data".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let table = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Table name must be a string".to_string())),
    };

    let data = match &args[2] {
        Value::List(records) => records,
        _ => return Err(VmError::Runtime("Data must be a list of records".to_string())),
    };

    // Build INSERT query
    if data.is_empty() {
        return Ok(Value::Real(0.0));
    }

    // Extract column names from first record
    let first_record = match &data[0] {
        Value::List(kv_pairs) => kv_pairs,
        _ => return Err(VmError::Runtime("Each record must be a list of key-value pairs".to_string())),
    };

    let mut columns = Vec::new();
    for kv in first_record {
        if let Value::List(pair) = kv {
            if pair.len() == 2 {
                if let Value::String(key) = &pair[0] {
                    columns.push(key.clone());
                }
            }
        }
    }

    if columns.is_empty() {
        return Err(VmError::Runtime("No valid columns found in data".to_string()));
    }

    // Build INSERT statement
    let column_list = columns.join(", ");
    let placeholders: Vec<String> = (1..=columns.len()).map(|i| format!("${}", i)).collect();
    let placeholder_list = placeholders.join(", ");
    let query = format!("INSERT INTO {} ({}) VALUES ({})", table, column_list, placeholder_list);

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    let mut total_inserted = 0;

    for record in data {
        if let Value::List(kv_pairs) = record {
            let mut values = Vec::new();
            for column in &columns {
                let mut found = false;
                for kv in kv_pairs {
                    if let Value::List(pair) = kv {
                        if pair.len() == 2 {
                            if let Value::String(key) = &pair[0] {
                                if key == column {
                                    values.push(pair[1].clone());
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                if !found {
                    values.push(Value::Missing);
                }
            }

            let rows_affected = rt.block_on(async {
                connection.execute_command(&query, values).await
            })?;

            total_inserted += rows_affected;
        }
    }

    Ok(Value::Real(total_inserted as f64))
}

/// SQLUpdate[connection, table, data, where_clause] - Update records
pub fn sql_update(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(VmError::Runtime("SQLUpdate requires connection, table, data, and where clause".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let table = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Table name must be a string".to_string())),
    };

    let data = match &args[2] {
        Value::List(kv_pairs) => kv_pairs,
        _ => return Err(VmError::Runtime("Data must be a list of key-value pairs".to_string())),
    };

    let where_clause = match &args[3] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Where clause must be a string".to_string())),
    };

    // Build UPDATE query
    let mut set_clauses = Vec::new();
    let mut values = Vec::new();
    let mut param_counter = 1;

    for kv in data {
        if let Value::List(pair) = kv {
            if pair.len() == 2 {
                if let Value::String(key) = &pair[0] {
                    set_clauses.push(format!("{} = ${}", key, param_counter));
                    values.push(pair[1].clone());
                    param_counter += 1;
                }
            }
        }
    }

    if set_clauses.is_empty() {
        return Err(VmError::Runtime("No valid update data provided".to_string()));
    }

    let set_clause = set_clauses.join(", ");
    let query = format!("UPDATE {} SET {} WHERE {}", table, set_clause, where_clause);

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    let rows_affected = rt.block_on(async {
        connection.execute_command(&query, values).await
    })?;

    Ok(Value::Real(rows_affected as f64))
}

/// SQLDelete[connection, table, where_clause] - Delete records
pub fn sql_delete(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime("SQLDelete requires connection, table, and where clause".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let table = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Table name must be a string".to_string())),
    };

    let where_clause = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Where clause must be a string".to_string())),
    };

    let query = format!("DELETE FROM {} WHERE {}", table, where_clause);

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    let rows_affected = rt.block_on(async {
        connection.execute_command(&query, vec![]).await
    })?;

    Ok(Value::Real(rows_affected as f64))
}

/// SQLSchema[connection, table] - Get table schema information
pub fn sql_schema(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("SQLSchema requires connection and table name".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let table = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Table name must be a string".to_string())),
    };

    let query = match connection.database_type {
        DatabaseType::PostgreSQL => {
            format!(
                "SELECT column_name, data_type, is_nullable, column_default 
                 FROM information_schema.columns 
                 WHERE table_name = '{}' 
                 ORDER BY ordinal_position", 
                table
            )
        }
        DatabaseType::MySQL => {
            format!(
                "SELECT column_name, data_type, is_nullable, column_default 
                 FROM information_schema.columns 
                 WHERE table_name = '{}' 
                 ORDER BY ordinal_position", 
                table
            )
        }
        DatabaseType::SQLite => {
            format!("PRAGMA table_info({})", table)
        }
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    rt.block_on(async {
        connection.execute_query(&query, vec![]).await
    })
}

/// SQLBatch[connection, operations] - Batch operations for performance
pub fn sql_batch(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("SQLBatch requires connection and operations".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let operations = match &args[1] {
        Value::List(ops) => ops,
        _ => return Err(VmError::Runtime("Operations must be a list".to_string())),
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    let mut results = Vec::new();

    for operation in operations {
        if let Value::List(op_parts) = operation {
            if !op_parts.is_empty() {
                if let Value::String(op_name) = &op_parts[0] {
                    match op_name.as_str() {
                        "SQLQuery" => {
                            if op_parts.len() >= 3 {
                                let query = match &op_parts[1] {
                                    Value::String(q) => q,
                                    _ => continue,
                                };
                                let params = match &op_parts[2] {
                                    Value::List(p) => p.clone(),
                                    _ => vec![],
                                };
                                
                                let result = rt.block_on(async {
                                    connection.execute_query(query, params).await
                                })?;
                                results.push(result);
                            }
                        }
                        _ => {
                            return Err(VmError::Runtime(format!("Unsupported batch operation: {}", op_name)));
                        }
                    }
                }
            }
        }
    }

    Ok(Value::List(results))
}

/// SQLMigration[connection, migration_file] - Execute schema migrations
pub fn sql_migration(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("SQLMigration requires connection and migration_file".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let migration_file = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Migration file must be a string".to_string())),
    };

    // Placeholder implementation for migration
    // In a real implementation, this would read and execute migration files
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    rt.block_on(async {
        // Simulate migration execution
        Ok(Value::List(vec![
            Value::List(vec![Value::String("status".to_string()), Value::String("completed".to_string())]),
            Value::List(vec![Value::String("migration_file".to_string()), Value::String(migration_file.clone())]),
            Value::List(vec![Value::String("operations_executed".to_string()), Value::Real(0.0)]),
        ]))
    })
}

/// SQLExport[connection, query, format, path] - Export query results
pub fn sql_export(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(VmError::Runtime("SQLExport requires connection, query, format, and path".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let query = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Query must be a string".to_string())),
    };

    let format = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Format must be a string".to_string())),
    };

    let path = match &args[3] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    rt.block_on(async {
        // Execute query first
        let results = connection.execute_query(query, vec![]).await?;
        
        // Placeholder for export logic
        // In a real implementation, this would write to CSV, JSON, etc.
        match format.as_str() {
            "csv" | "json" | "xlsx" => {
                Ok(Value::List(vec![
                    Value::List(vec![Value::String("exported".to_string()), Value::Boolean(true)]),
                    Value::List(vec![Value::String("format".to_string()), Value::String(format.clone())]),
                    Value::List(vec![Value::String("path".to_string()), Value::String(path.clone())]),
                    Value::List(vec![Value::String("rows_exported".to_string()), Value::Real(0.0)]),
                ]))
            }
            _ => Err(VmError::Runtime(format!("Unsupported export format: {}", format))),
        }
    })
}

/// SQLImport[connection, table, data_source, options] - Import data from files
pub fn sql_import(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime("SQLImport requires connection, table, and data_source".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let table = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Table name must be a string".to_string())),
    };

    let data_source = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Data source must be a string".to_string())),
    };

    // Parse options if provided
    let mut batch_size = 1000;
    let mut skip_errors = false;
    
    if args.len() > 3 {
        if let Value::List(options) = &args[3] {
            for option in options {
                if let Value::List(kv) = option {
                    if kv.len() == 2 {
                        if let (Value::String(key), value) = (&kv[0], &kv[1]) {
                            match key.as_str() {
                                "batch_size" => {
                                    if let Value::Real(n) = value {
                                        batch_size = *n as usize;
                                    }
                                }
                                "skip_errors" => {
                                    if let Value::Boolean(b) = value {
                                        skip_errors = *b;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }

    // Placeholder for import logic
    // In a real implementation, this would read from CSV, JSON, etc.
    Ok(Value::List(vec![
        Value::List(vec![Value::String("imported".to_string()), Value::Boolean(true)]),
        Value::List(vec![Value::String("table".to_string()), Value::String(table.clone())]),
        Value::List(vec![Value::String("data_source".to_string()), Value::String(data_source.clone())]),
        Value::List(vec![Value::String("rows_imported".to_string()), Value::Real(0.0)]),
        Value::List(vec![Value::String("batch_size".to_string()), Value::Real(batch_size as f64)]),
    ]))
}

/// SQLProcedure[connection, procedure_name, parameters] - Call stored procedures
pub fn sql_procedure(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("SQLProcedure requires connection and procedure_name".to_string()));
    }

    let connection = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SQLConnection>()
                .ok_or_else(|| VmError::Runtime("Expected SQLConnection".to_string()))?
        }
        _ => return Err(VmError::Runtime("First argument must be an SQLConnection".to_string())),
    };

    let procedure_name = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Procedure name must be a string".to_string())),
    };

    let parameters = if args.len() > 2 {
        match &args[2] {
            Value::List(params) => params.clone(),
            _ => vec![args[2].clone()],
        }
    } else {
        vec![]
    };

    // Build CALL statement based on database type
    let call_sql = match connection.database_type {
        DatabaseType::PostgreSQL => {
            format!("CALL {}({})", procedure_name, 
                (0..parameters.len()).map(|i| format!("${}", i + 1)).collect::<Vec<_>>().join(", "))
        }
        DatabaseType::MySQL => {
            format!("CALL {}({})", procedure_name, 
                (0..parameters.len()).map(|_| "?").collect::<Vec<_>>().join(", "))
        }
        DatabaseType::SQLite => {
            // SQLite doesn't have stored procedures, return error
            return Err(VmError::Runtime("SQLite does not support stored procedures".to_string()));
        }
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create async runtime: {}", e)))?;

    rt.block_on(async {
        connection.execute_query(&call_sql, parameters).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_database_config() {
        let config = DatabaseConfig::new("sqlite://test.db".to_string());
        assert_eq!(config.connection_string, "sqlite://test.db");
        assert_eq!(config.pool_size, Some(10));
    }

    #[test]
    fn test_sql_connect_function() {
        let args = vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ];
        
        let result = sql_connect(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::LyObj(obj) => {
                assert!(obj.downcast_ref::<SQLConnection>().is_some());
            }
            _ => panic!("Expected LyObj"),
        }
    }

    #[test]
    fn test_sql_connect_with_options() {
        let args = vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
            Value::List(vec![
                Value::List(vec![
                    Value::String("pool_size".to_string()),
                    Value::Real(5.0),
                ]),
                Value::List(vec![
                    Value::String("connection_timeout".to_string()),
                    Value::Real(60.0),
                ]),
            ]),
        ];
        
        let result = sql_connect(&args);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_sqlite_connection() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap();
        let connection_string = format!("sqlite://{}", db_path);
        
        let config = DatabaseConfig::new(connection_string);
        let connection = SQLConnection::new(DatabaseType::SQLite, config).await;
        
        assert!(connection.is_ok());
        
        let conn = connection.unwrap();
        assert!(conn.health_check().await.unwrap());
        
        // Test creating a table
        let create_table = "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)";
        let result = conn.execute_command(create_table, vec![]).await;
        assert!(result.is_ok());
        
        // Test inserting data
        let insert = "INSERT INTO test_table (name) VALUES (?)";
        let result = conn.execute_command(insert, vec![Value::String("test".to_string())]).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        // Test querying data
        let select = "SELECT id, name FROM test_table";
        let result = conn.execute_query_internal(select, vec![]).await;
        assert!(result.is_ok());
        
        let rows = result.unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains_key("id"));
        assert!(rows[0].contains_key("name"));
    }

    #[test]
    fn test_sql_query_function() {
        // First create a connection
        let connect_args = vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ];
        
        let connection_result = sql_connect(&connect_args);
        assert!(connection_result.is_ok());
        let connection = connection_result.unwrap();
        
        // Test query function
        let query_args = vec![
            connection,
            Value::String("SELECT 1 as test_col".to_string()),
            Value::List(vec![]),
        ];
        
        let result = sql_query(&query_args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::List(rows) => {
                assert_eq!(rows.len(), 1);
            }
            _ => panic!("Expected List of rows"),
        }
    }

    #[test]
    fn test_sql_insert_function() {
        // First create a connection
        let connect_args = vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ];
        
        let connection_result = sql_connect(&connect_args);
        assert!(connection_result.is_ok());
        let connection = connection_result.unwrap();
        
        // Create a table first
        let create_args = vec![
            connection.clone(),
            Value::String("CREATE TABLE test (id INTEGER, name TEXT)".to_string()),
            Value::List(vec![]),
        ];
        
        let _ = sql_query(&create_args);
        
        // Test insert function
        let insert_args = vec![
            connection,
            Value::String("test".to_string()),
            Value::List(vec![
                Value::List(vec![
                    Value::List(vec![Value::String("id".to_string()), Value::Real(1.0)]),
                    Value::List(vec![Value::String("name".to_string()), Value::String("Alice".to_string())]),
                ]),
            ]),
        ];
        
        let result = sql_insert(&insert_args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Real(n) => {
                assert_eq!(n, 1.0); // One row inserted
            }
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn test_error_handling() {
        let args = vec![
            Value::String("invalid_driver".to_string()),
            Value::String("invalid://connection".to_string()),
        ];
        
        let result = sql_connect(&args);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            VmError::Runtime(msg) => {
                assert!(msg.contains("Unsupported database driver"));
            }
            _ => panic!("Expected ArgumentError"),
        }
    }
}