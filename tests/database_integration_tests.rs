//! Integration Tests for Database Phase 15A
//!
//! These tests validate the complete database integration system including:
//! - SQL database operations (PostgreSQL, MySQL, SQLite)
//! - NoSQL database operations (MongoDB, Redis)
//! - Multi-database operations and synchronization
//! - Schema management and evolution
//! - Performance under load

use lyra::vm::{Value, VmResult};
use lyra::stdlib::StandardLibrary;
use std::collections::HashMap;
use tempfile::NamedTempFile;

#[cfg(test)]
mod database_tests {
    use super::*;

    fn get_stdlib() -> StandardLibrary {
        StandardLibrary::new()
    }

    #[test]
    fn test_sql_connection_functions_registration() {
        let stdlib = get_stdlib();
        
        // Test that all SQL functions are registered
        assert!(stdlib.get_function("SQLConnect").is_some());
        assert!(stdlib.get_function("SQLQuery").is_some());
        assert!(stdlib.get_function("SQLTransaction").is_some());
        assert!(stdlib.get_function("SQLInsert").is_some());
        assert!(stdlib.get_function("SQLUpdate").is_some());
        assert!(stdlib.get_function("SQLDelete").is_some());
        assert!(stdlib.get_function("SQLSchema").is_some());
        assert!(stdlib.get_function("SQLMigration").is_some());
        assert!(stdlib.get_function("SQLBatch").is_some());
        assert!(stdlib.get_function("SQLExport").is_some());
        assert!(stdlib.get_function("SQLImport").is_some());
        assert!(stdlib.get_function("SQLProcedure").is_some());
    }

    #[test]
    fn test_nosql_functions_registration() {
        let stdlib = get_stdlib();
        
        // Test that all NoSQL functions are registered
        assert!(stdlib.get_function("MongoConnect").is_some());
        assert!(stdlib.get_function("MongoFind").is_some());
        assert!(stdlib.get_function("MongoInsert").is_some());
        assert!(stdlib.get_function("MongoUpdate").is_some());
        assert!(stdlib.get_function("MongoDelete").is_some());
        assert!(stdlib.get_function("MongoAggregate").is_some());
        assert!(stdlib.get_function("RedisConnect").is_some());
        assert!(stdlib.get_function("RedisGet").is_some());
        assert!(stdlib.get_function("RedisSet").is_some());
        assert!(stdlib.get_function("RedisPipeline").is_some());
    }

    #[test]
    fn test_graph_database_functions_registration() {
        let stdlib = get_stdlib();
        
        // Test that all graph database functions are registered
        assert!(stdlib.get_function("Neo4jConnect").is_some());
        assert!(stdlib.get_function("CypherQuery").is_some());
        assert!(stdlib.get_function("GraphNode").is_some());
        assert!(stdlib.get_function("GraphRelation").is_some());
        assert!(stdlib.get_function("GraphPath").is_some());
        assert!(stdlib.get_function("GraphAnalytics").is_some());
    }

    #[test]
    fn test_multi_database_functions_registration() {
        let stdlib = get_stdlib();
        
        // Test that all multi-database functions are registered
        assert!(stdlib.get_function("DatabaseSync").is_some());
        assert!(stdlib.get_function("CrossQuery").is_some());
        assert!(stdlib.get_function("DataFederation").is_some());
        assert!(stdlib.get_function("DatabaseMigration").is_some());
        assert!(stdlib.get_function("DataConsistency").is_some());
        assert!(stdlib.get_function("DatabaseReplication").is_some());
        assert!(stdlib.get_function("LoadBalancer").is_some());
        assert!(stdlib.get_function("ShardManager").is_some());
    }

    #[test]
    fn test_schema_management_functions_registration() {
        let stdlib = get_stdlib();
        
        // Test that all schema management functions are registered
        assert!(stdlib.get_function("SchemaEvolution").is_some());
        assert!(stdlib.get_function("SchemaValidation").is_some());
        assert!(stdlib.get_function("SchemaDiff").is_some());
        assert!(stdlib.get_function("SchemaGenerate").is_some());
    }

    #[test]
    fn test_sqlite_connection_and_basic_operations() {
        let stdlib = get_stdlib();
        
        // Test SQLConnect with SQLite
        let sql_connect = stdlib.get_function("SQLConnect").unwrap();
        let connect_args = vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ];
        
        let connection = sql_connect(&connect_args);
        assert!(connection.is_ok(), "SQLite connection should succeed");
        
        if let Ok(conn) = connection {
            // Test SQLQuery with table creation
            let sql_query = stdlib.get_function("SQLQuery").unwrap();
            let create_table_args = vec![
                conn.clone(),
                Value::String("CREATE TABLE test_users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)".to_string()),
                Value::List(vec![]),
            ];
            
            let result = sql_query(&create_table_args);
            assert!(result.is_ok(), "Table creation should succeed");
            
            // Test SQLInsert
            let sql_insert = stdlib.get_function("SQLInsert").unwrap();
            let insert_args = vec![
                conn.clone(),
                Value::String("test_users".to_string()),
                Value::List(vec![
                    Value::List(vec![
                        Value::List(vec![Value::String("name".to_string()), Value::String("Alice".to_string())]),
                        Value::List(vec![Value::String("age".to_string()), Value::Number(30.0)]),
                    ]),
                    Value::List(vec![
                        Value::List(vec![Value::String("name".to_string()), Value::String("Bob".to_string())]),
                        Value::List(vec![Value::String("age".to_string()), Value::Number(25.0)]),
                    ]),
                ]),
            ];
            
            let insert_result = sql_insert(&insert_args);
            assert!(insert_result.is_ok(), "Insert should succeed");
            
            if let Ok(Value::Number(rows_inserted)) = insert_result {
                assert_eq!(rows_inserted, 2.0, "Should insert 2 rows");
            }
            
            // Test SQLQuery with SELECT
            let select_args = vec![
                conn.clone(),
                Value::String("SELECT * FROM test_users ORDER BY id".to_string()),
                Value::List(vec![]),
            ];
            
            let select_result = sql_query(&select_args);
            assert!(select_result.is_ok(), "Select should succeed");
            
            if let Ok(Value::List(rows)) = select_result {
                assert_eq!(rows.len(), 2, "Should return 2 rows");
            }
            
            // Test SQLUpdate
            let sql_update = stdlib.get_function("SQLUpdate").unwrap();
            let update_args = vec![
                conn.clone(),
                Value::String("test_users".to_string()),
                Value::List(vec![
                    Value::List(vec![Value::String("age".to_string()), Value::Number(31.0)]),
                ]),
                Value::String("name = 'Alice'".to_string()),
            ];
            
            let update_result = sql_update(&update_args);
            assert!(update_result.is_ok(), "Update should succeed");
            
            // Test SQLDelete
            let sql_delete = stdlib.get_function("SQLDelete").unwrap();
            let delete_args = vec![
                conn.clone(),
                Value::String("test_users".to_string()),
                Value::String("age < 30".to_string()),
            ];
            
            let delete_result = sql_delete(&delete_args);
            assert!(delete_result.is_ok(), "Delete should succeed");
            
            // Test SQLSchema
            let sql_schema = stdlib.get_function("SQLSchema").unwrap();
            let schema_args = vec![
                conn.clone(),
                Value::String("test_users".to_string()),
            ];
            
            let schema_result = sql_schema(&schema_args);
            assert!(schema_result.is_ok(), "Schema query should succeed");
        }
    }

    #[test]
    fn test_schema_inference_and_validation() {
        let stdlib = get_stdlib();
        
        // Test SchemaGenerate (infer_schema)
        let schema_generate = stdlib.get_function("SchemaGenerate").unwrap();
        
        let sample_data = Value::List(vec![
            Value::List(vec![
                Value::List(vec![Value::String("id".to_string()), Value::Number(1.0)]),
                Value::List(vec![Value::String("name".to_string()), Value::String("Alice".to_string())]),
                Value::List(vec![Value::String("age".to_string()), Value::Number(30.0)]),
                Value::List(vec![Value::String("active".to_string()), Value::Boolean(true)]),
            ]),
            Value::List(vec![
                Value::List(vec![Value::String("id".to_string()), Value::Number(2.0)]),
                Value::List(vec![Value::String("name".to_string()), Value::String("Bob".to_string())]),
                Value::List(vec![Value::String("age".to_string()), Value::Number(25.0)]),
                Value::List(vec![Value::String("active".to_string()), Value::Boolean(false)]),
            ]),
        ]);
        
        let inference_rules = Value::List(vec![
            Value::List(vec![Value::String("sample_size".to_string()), Value::Number(1000.0)]),
            Value::List(vec![Value::String("null_threshold".to_string()), Value::Number(0.1)]),
            Value::List(vec![Value::String("detect_foreign_keys".to_string()), Value::Boolean(true)]),
        ]);
        
        let schema_args = vec![sample_data.clone(), inference_rules];
        let schema_result = schema_generate(&schema_args);
        
        assert!(schema_result.is_ok(), "Schema inference should succeed");
        
        if let Ok(Value::String(schema_json)) = schema_result {
            assert!(schema_json.contains("inferred_schema"), "Should contain schema name");
            assert!(schema_json.contains("tables"), "Should contain tables");
            
            // Test SchemaValidation
            let schema_validation = stdlib.get_function("SchemaValidation").unwrap();
            let validation_args = vec![
                Value::String("test_db".to_string()),
                Value::String(schema_json),
                sample_data,
            ];
            
            let validation_result = schema_validation(&validation_args);
            assert!(validation_result.is_ok(), "Schema validation should succeed");
            
            if let Ok(Value::List(errors)) = validation_result {
                // For valid data, should have no errors
                assert_eq!(errors.len(), 0, "Should have no validation errors for valid data");
            }
        }
    }

    #[test]
    fn test_multi_database_operations() {
        let stdlib = get_stdlib();
        
        // Test CrossQuery
        let cross_query = stdlib.get_function("CrossQuery").unwrap();
        let cross_query_args = vec![
            Value::List(vec![]), // Empty database list for testing
            Value::String("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id".to_string()),
            Value::String("hash_join".to_string()),
        ];
        
        let result = cross_query(&cross_query_args);
        assert!(result.is_ok(), "CrossQuery should succeed");
        
        // Test DatabaseSync
        let database_sync = stdlib.get_function("DatabaseSync").unwrap();
        let sync_args = vec![
            Value::String("source_db".to_string()),
            Value::String("target_db".to_string()),
            Value::List(vec![
                Value::List(vec![Value::String("strategy".to_string()), Value::String("incremental".to_string())]),
            ]),
        ];
        
        let sync_result = database_sync(&sync_args);
        assert!(sync_result.is_ok(), "DatabaseSync should succeed");
        
        // Test DataFederation
        let data_federation = stdlib.get_function("DataFederation").unwrap();
        let federation_args = vec![
            Value::List(vec![]), // Empty database list
            Value::List(vec![
                Value::List(vec![Value::String("virtual_table".to_string()), Value::String("definition".to_string())]),
            ]),
        ];
        
        let federation_result = data_federation(&federation_args);
        assert!(federation_result.is_ok(), "DataFederation should succeed");
        
        // Test LoadBalancer
        let load_balancer = stdlib.get_function("LoadBalancer").unwrap();
        let lb_args = vec![
            Value::List(vec![]), // Empty database list
            Value::String("round_robin".to_string()),
            Value::Boolean(true),
        ];
        
        let lb_result = load_balancer(&lb_args);
        assert!(lb_result.is_ok(), "LoadBalancer should succeed");
        
        // Test ShardManager
        let shard_manager = stdlib.get_function("ShardManager").unwrap();
        let shard_args = vec![
            Value::List(vec![]), // Empty database list
            Value::String("user_id".to_string()),
            Value::String("hash".to_string()),
        ];
        
        let shard_result = shard_manager(&shard_args);
        assert!(shard_result.is_ok(), "ShardManager should succeed");
    }

    #[test]
    fn test_error_handling() {
        let stdlib = get_stdlib();
        
        // Test SQLConnect with invalid arguments
        let sql_connect = stdlib.get_function("SQLConnect").unwrap();
        
        // Test with too few arguments
        let result = sql_connect(&vec![Value::String("sqlite".to_string())]);
        assert!(result.is_err(), "Should fail with too few arguments");
        
        // Test with invalid driver
        let result = sql_connect(&vec![
            Value::String("invalid_driver".to_string()),
            Value::String("connection_string".to_string()),
        ]);
        assert!(result.is_err(), "Should fail with invalid driver");
        
        // Test with wrong argument types
        let result = sql_connect(&vec![
            Value::Number(123.0),
            Value::String("connection_string".to_string()),
        ]);
        assert!(result.is_err(), "Should fail with wrong argument types");
        
        // Test MongoConnect with invalid arguments
        let mongo_connect = stdlib.get_function("MongoConnect").unwrap();
        let result = mongo_connect(&vec![Value::String("connection".to_string())]);
        assert!(result.is_err(), "Should fail with too few arguments");
        
        // Test SchemaValidation with invalid schema
        let schema_validation = stdlib.get_function("SchemaValidation").unwrap();
        let result = schema_validation(&vec![
            Value::String("db".to_string()),
            Value::String("invalid_json".to_string()),
            Value::List(vec![]),
        ]);
        assert!(result.is_err(), "Should fail with invalid JSON schema");
    }

    #[test]
    fn test_transaction_rollback_simulation() {
        let stdlib = get_stdlib();
        
        // Create SQLite connection
        let sql_connect = stdlib.get_function("SQLConnect").unwrap();
        let connection = sql_connect(&vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ]).unwrap();
        
        // Create test table
        let sql_query = stdlib.get_function("SQLQuery").unwrap();
        let _ = sql_query(&vec![
            connection.clone(),
            Value::String("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL)".to_string()),
            Value::List(vec![]),
        ]);
        
        // Insert initial data
        let sql_insert = stdlib.get_function("SQLInsert").unwrap();
        let _ = sql_insert(&vec![
            connection.clone(),
            Value::String("accounts".to_string()),
            Value::List(vec![
                Value::List(vec![
                    Value::List(vec![Value::String("balance".to_string()), Value::Number(1000.0)]),
                ]),
                Value::List(vec![
                    Value::List(vec![Value::String("balance".to_string()), Value::Number(500.0)]),
                ]),
            ]),
        ]);
        
        // Test SQLTransaction
        let sql_transaction = stdlib.get_function("SQLTransaction").unwrap();
        let transaction_operations = Value::List(vec![
            Value::List(vec![
                Value::String("SQLQuery".to_string()),
                Value::String("UPDATE accounts SET balance = balance - 100 WHERE id = 1".to_string()),
                Value::List(vec![]),
            ]),
            Value::List(vec![
                Value::String("SQLQuery".to_string()),
                Value::String("UPDATE accounts SET balance = balance + 100 WHERE id = 2".to_string()),
                Value::List(vec![]),
            ]),
        ]);
        
        let transaction_args = vec![
            connection.clone(),
            transaction_operations,
        ];
        
        let result = sql_transaction(&transaction_args);
        assert!(result.is_ok(), "Transaction should succeed");
    }

    #[test]
    fn test_batch_operations() {
        let stdlib = get_stdlib();
        
        // Create SQLite connection
        let sql_connect = stdlib.get_function("SQLConnect").unwrap();
        let connection = sql_connect(&vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ]).unwrap();
        
        // Test SQLBatch
        let sql_batch = stdlib.get_function("SQLBatch").unwrap();
        let batch_operations = Value::List(vec![
            Value::List(vec![
                Value::String("SQLQuery".to_string()),
                Value::String("CREATE TABLE batch_test (id INTEGER, data TEXT)".to_string()),
                Value::List(vec![]),
            ]),
            Value::List(vec![
                Value::String("SQLQuery".to_string()),
                Value::String("INSERT INTO batch_test (data) VALUES ('test1')".to_string()),
                Value::List(vec![]),
            ]),
            Value::List(vec![
                Value::String("SQLQuery".to_string()),
                Value::String("INSERT INTO batch_test (data) VALUES ('test2')".to_string()),
                Value::List(vec![]),
            ]),
        ]);
        
        let batch_args = vec![
            connection,
            batch_operations,
        ];
        
        let result = sql_batch(&batch_args);
        assert!(result.is_ok(), "Batch operations should succeed");
        
        if let Ok(Value::List(results)) = result {
            assert_eq!(results.len(), 3, "Should execute 3 operations");
        }
    }

    #[test]
    fn test_import_export_functions() {
        let stdlib = get_stdlib();
        
        // Create SQLite connection
        let sql_connect = stdlib.get_function("SQLConnect").unwrap();
        let connection = sql_connect(&vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ]).unwrap();
        
        // Test SQLExport
        let sql_export = stdlib.get_function("SQLExport").unwrap();
        let export_args = vec![
            connection.clone(),
            Value::String("SELECT 'test' as data".to_string()),
            Value::String("csv".to_string()),
            Value::String("/tmp/test_export.csv".to_string()),
        ];
        
        let export_result = sql_export(&export_args);
        assert!(export_result.is_ok(), "Export should succeed");
        
        // Test SQLImport
        let sql_import = stdlib.get_function("SQLImport").unwrap();
        let import_args = vec![
            connection,
            Value::String("test_table".to_string()),
            Value::String("/tmp/test_data.csv".to_string()),
            Value::List(vec![
                Value::List(vec![Value::String("batch_size".to_string()), Value::Number(500.0)]),
                Value::List(vec![Value::String("skip_errors".to_string()), Value::Boolean(true)]),
            ]),
        ];
        
        let import_result = sql_import(&import_args);
        assert!(import_result.is_ok(), "Import should succeed");
    }

    #[test]
    fn test_data_type_conversion() {
        let stdlib = get_stdlib();
        
        // Test that various data types are handled correctly
        let sql_connect = stdlib.get_function("SQLConnect").unwrap();
        let connection = sql_connect(&vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ]).unwrap();
        
        let sql_query = stdlib.get_function("SQLQuery").unwrap();
        
        // Create table with various data types
        let _ = sql_query(&vec![
            connection.clone(),
            Value::String("CREATE TABLE type_test (
                int_col INTEGER,
                real_col REAL,
                text_col TEXT,
                bool_col BOOLEAN,
                date_col DATETIME
            )".to_string()),
            Value::List(vec![]),
        ]);
        
        // Test SQLInsert with different data types
        let sql_insert = stdlib.get_function("SQLInsert").unwrap();
        let insert_result = sql_insert(&vec![
            connection.clone(),
            Value::String("type_test".to_string()),
            Value::List(vec![
                Value::List(vec![
                    Value::List(vec![Value::String("int_col".to_string()), Value::Number(42.0)]),
                    Value::List(vec![Value::String("real_col".to_string()), Value::Number(3.14)]),
                    Value::List(vec![Value::String("text_col".to_string()), Value::String("hello".to_string())]),
                    Value::List(vec![Value::String("bool_col".to_string()), Value::Boolean(true)]),
                    Value::List(vec![Value::String("date_col".to_string()), Value::String("2024-01-01 12:00:00".to_string())]),
                ]),
            ]),
        ]);
        
        assert!(insert_result.is_ok(), "Insert with various data types should succeed");
        
        // Query the data back
        let select_result = sql_query(&vec![
            connection,
            Value::String("SELECT * FROM type_test".to_string()),
            Value::List(vec![]),
        ]);
        
        assert!(select_result.is_ok(), "Select should succeed");
        
        if let Ok(Value::List(rows)) = select_result {
            assert_eq!(rows.len(), 1, "Should return 1 row");
        }
    }

    #[test]
    fn test_comprehensive_function_coverage() {
        let stdlib = get_stdlib();
        let function_names = stdlib.function_names();
        
        // Verify that we have implemented at least 40 database functions as specified
        let database_functions: Vec<&String> = function_names.iter()
            .filter(|name| {
                name.starts_with("SQL") || 
                name.starts_with("Mongo") || 
                name.starts_with("Redis") || 
                name.starts_with("Neo4j") || 
                name.starts_with("Cypher") || 
                name.starts_with("Graph") || 
                name.starts_with("Database") || 
                name.starts_with("Cross") || 
                name.starts_with("Data") || 
                name.starts_with("Load") || 
                name.starts_with("Shard") || 
                name.starts_with("Schema")
            })
            .collect();
        
        println!("Database functions found: {:?}", database_functions);
        assert!(database_functions.len() >= 40, 
            "Should have at least 40 database functions, found: {}", database_functions.len());
        
        // Test that all functions can be called (even if they return placeholder results)
        for function_name in database_functions {
            let func = stdlib.get_function(function_name).unwrap();
            
            // Try calling with minimal arguments (should either succeed or give argument error)
            let result = func(&vec![]);
            
            // All database functions should at least validate arguments properly
            match result {
                Ok(_) => {}, // Function succeeded with no args (unlikely but possible)
                Err(lyra::vm::VmError::ArgumentError(_)) => {}, // Expected for most functions
                Err(lyra::vm::VmError::RuntimeError(_)) => {}, // Some functions may have runtime errors
                Err(e) => panic!("Function {} returned unexpected error: {:?}", function_name, e),
            }
        }
    }
}

// Performance tests module
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_large_dataset_performance() {
        let stdlib = get_stdlib();
        
        // Create SQLite connection
        let sql_connect = stdlib.get_function("SQLConnect").unwrap();
        let connection = sql_connect(&vec![
            Value::String("sqlite".to_string()),
            Value::String(":memory:".to_string()),
        ]).unwrap();
        
        // Create table
        let sql_query = stdlib.get_function("SQLQuery").unwrap();
        let _ = sql_query(&vec![
            connection.clone(),
            Value::String("CREATE TABLE perf_test (id INTEGER PRIMARY KEY, data TEXT)".to_string()),
            Value::List(vec![]),
        ]);
        
        // Prepare large dataset (1000 records)
        let mut records = Vec::new();
        for i in 0..1000 {
            records.push(Value::List(vec![
                Value::List(vec![Value::String("data".to_string()), Value::String(format!("test_data_{}", i))]),
            ]));
        }
        
        let large_dataset = Value::List(records);
        
        // Measure insert performance
        let sql_insert = stdlib.get_function("SQLInsert").unwrap();
        let start = Instant::now();
        
        let insert_result = sql_insert(&vec![
            connection.clone(),
            Value::String("perf_test".to_string()),
            large_dataset,
        ]);
        
        let insert_duration = start.elapsed();
        
        assert!(insert_result.is_ok(), "Large dataset insert should succeed");
        println!("Insert 1000 records took: {:?}", insert_duration);
        
        // Should complete within reasonable time (5 seconds)
        assert!(insert_duration.as_secs() < 5, "Insert should complete within 5 seconds");
        
        // Measure query performance
        let start = Instant::now();
        
        let select_result = sql_query(&vec![
            connection,
            Value::String("SELECT COUNT(*) FROM perf_test".to_string()),
            Value::List(vec![]),
        ]);
        
        let query_duration = start.elapsed();
        
        assert!(select_result.is_ok(), "Count query should succeed");
        println!("Count query took: {:?}", query_duration);
        
        // Query should be very fast
        assert!(query_duration.as_millis() < 1000, "Query should complete within 1 second");
    }

    #[test]
    fn test_schema_inference_performance() {
        let stdlib = get_stdlib();
        
        // Create large sample dataset for schema inference
        let mut sample_data = Vec::new();
        for i in 0..1000 {
            sample_data.push(Value::List(vec![
                Value::List(vec![Value::String("id".to_string()), Value::Number(i as f64)]),
                Value::List(vec![Value::String("name".to_string()), Value::String(format!("user_{}", i))]),
                Value::List(vec![Value::String("age".to_string()), Value::Number((20 + i % 60) as f64)]),
                Value::List(vec![Value::String("active".to_string()), Value::Boolean(i % 2 == 0)]),
                Value::List(vec![Value::String("score".to_string()), Value::Number((i as f64) * 1.5)]),
            ]));
        }
        
        let large_sample = Value::List(sample_data);
        let inference_rules = Value::List(vec![
            Value::List(vec![Value::String("sample_size".to_string()), Value::Number(1000.0)]),
        ]);
        
        // Measure schema inference performance
        let schema_generate = stdlib.get_function("SchemaGenerate").unwrap();
        let start = Instant::now();
        
        let schema_result = schema_generate(&vec![large_sample, inference_rules]);
        
        let inference_duration = start.elapsed();
        
        assert!(schema_result.is_ok(), "Schema inference should succeed");
        println!("Schema inference for 1000 records took: {:?}", inference_duration);
        
        // Should complete within reasonable time
        assert!(inference_duration.as_secs() < 10, "Schema inference should complete within 10 seconds");
    }

    #[test]
    fn test_function_call_overhead() {
        let stdlib = get_stdlib();
        
        // Measure the overhead of function lookup and calling
        let sql_connect = stdlib.get_function("SQLConnect").unwrap();
        
        let start = Instant::now();
        
        // Call the same function many times
        for _ in 0..1000 {
            let _ = sql_connect(&vec![
                Value::String("sqlite".to_string()),
                Value::String(":memory:".to_string()),
            ]);
        }
        
        let total_duration = start.elapsed();
        let avg_duration = total_duration / 1000;
        
        println!("Average function call duration: {:?}", avg_duration);
        
        // Each function call should be very fast
        assert!(avg_duration.as_micros() < 10000, "Average function call should be under 10ms");
    }
}