//! Benchmarks for the Lyra serialization system

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use lyra::serialization::{
    Serializable, BinaryWriter, BinaryReader, compression::{CompressionConfig, CompressionAlgorithm, SmartCompressor},
    session::{SessionState, SessionManager, SessionConfig, SimpleValue},
    zero_copy::{MappedFile, ZeroCopyDeserializer, StreamingDeserializer},
};
use lyra::ast::{Number, Symbol};
use std::collections::HashMap;
use std::io::Cursor;
use tempfile::NamedTempFile;

fn bench_basic_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_serialization");
    
    // Test different data sizes
    let sizes = vec![100, 1000, 10000, 100000];
    
    for size in sizes {
        // Create test data
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("string", size), &size, |b, &size| {
            let test_string = "a".repeat(size);
            b.iter(|| {
                let mut buffer = Vec::new();
                test_string.serialize(&mut buffer).unwrap();
                
                let mut cursor = Cursor::new(buffer);
                let _deserialized = String::deserialize(&mut cursor).unwrap();
            });
        });
        
        group.bench_with_input(BenchmarkId::new("number", size), &size, |b, &_size| {
            let numbers: Vec<Number> = (0..size).map(|i| {
                if i % 2 == 0 {
                    Number::Integer(i as i64)
                } else {
                    Number::Real(i as f64 + 0.5)
                }
            }).collect();
            
            b.iter(|| {
                for number in &numbers {
                    let mut buffer = Vec::new();
                    number.serialize(&mut buffer).unwrap();
                    
                    let mut cursor = Cursor::new(buffer);
                    let _deserialized = Number::deserialize(&mut cursor).unwrap();
                }
            });
        });
    }
    
    group.finish();
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");
    
    // Create test data of different characteristics
    let text_data = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(1000);
    let random_data: Vec<u8> = (0..50000).map(|i| (i * 7919) as u8).collect();
    let repetitive_data = vec![42u8; 50000];
    
    let datasets = vec![
        ("text", text_data.as_bytes()),
        ("random", &random_data),
        ("repetitive", &repetitive_data),
    ];
    
    let algorithms = vec![
        ("none", CompressionAlgorithm::None),
        ("lz4", CompressionAlgorithm::Lz4),
        ("zstd", CompressionAlgorithm::Zstd),
    ];
    
    for (data_name, data) in &datasets {
        group.throughput(Throughput::Bytes(data.len() as u64));
        
        for (algo_name, algorithm) in &algorithms {
            let config = CompressionConfig {
                algorithm: *algorithm,
                level: 1,
                dictionary: None,
            };
            
            group.bench_with_input(
                BenchmarkId::new(format!("compress_{}_{}", data_name, algo_name), data.len()),
                &(data, config),
                |b, &(data, config)| {
                    b.iter(|| {
                        let _compressed = lyra::serialization::compression::compress(black_box(data), black_box(&config)).unwrap();
                    });
                }
            );
        }
    }
    
    // Smart compression benchmark
    let compressor = SmartCompressor::new();
    for (data_name, data) in &datasets {
        group.bench_with_input(
            BenchmarkId::new(format!("smart_compress_{}", data_name), data.len()),
            data,
            |b, &data| {
                b.iter(|| {
                    let _compressed = compressor.compress_adaptive(black_box(data)).unwrap();
                });
            }
        );
    }
    
    group.finish();
}

fn bench_session_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("session_persistence");
    
    // Create test sessions of different sizes
    let session_sizes = vec![10, 100, 1000];
    
    for size in session_sizes {
        let mut state = SessionState::new();
        
        // Add variables
        for i in 0..size {
            let name = format!("var_{}", i);
            let value = match i % 4 {
                0 => SimpleValue::Integer(i as i64),
                1 => SimpleValue::Real(i as f64 + 0.5),
                2 => SimpleValue::String(format!("string_{}", i)),
                _ => SimpleValue::List(vec![
                    SimpleValue::Integer(i as i64),
                    SimpleValue::Integer((i + 1) as i64),
                ]),
            };
            state.add_variable(name, value);
        }
        
        // Add history
        for i in 0..size {
            state.add_history_entry(format!("command_{} = {}", i, i));
        }
        
        let estimated_size = state.serialized_size();
        group.throughput(Throughput::Bytes(estimated_size as u64));
        
        // Benchmark session save/load
        group.bench_with_input(
            BenchmarkId::new("save_load_compressed", size),
            &state,
            |b, state| {
                b.iter(|| {
                    let temp_file = NamedTempFile::new().unwrap();
                    let config = SessionConfig {
                        session_file: temp_file.path().to_path_buf(),
                        compression: true,
                        ..Default::default()
                    };
                    
                    let mut manager = SessionManager::new(config);
                    manager.save_session(black_box(state)).unwrap();
                    let _loaded = manager.load_session().unwrap();
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("save_load_uncompressed", size),
            &state,
            |b, state| {
                b.iter(|| {
                    let temp_file = NamedTempFile::new().unwrap();
                    let config = SessionConfig {
                        session_file: temp_file.path().to_path_buf(),
                        compression: false,
                        ..Default::default()
                    };
                    
                    let mut manager = SessionManager::new(config);
                    manager.save_session(black_box(state)).unwrap();
                    let _loaded = manager.load_session().unwrap();
                });
            }
        );
    }
    
    group.finish();
}

fn bench_zero_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy");
    
    // Create test data
    let test_data = {
        let mut data = Vec::new();
        data.extend_from_slice(&lyra::serialization::LYRA_MAGIC);
        data.extend_from_slice(&lyra::serialization::SERIALIZATION_VERSION.to_le_bytes());
        
        // Add some test payload
        for i in 0..10000u32 {
            data.extend_from_slice(&i.to_le_bytes());
        }
        data
    };
    
    // Write to temporary file
    let temp_file = NamedTempFile::new().unwrap();
    std::fs::write(temp_file.path(), &test_data).unwrap();
    
    group.throughput(Throughput::Bytes(test_data.len() as u64));
    
    group.bench_function("memory_map_file", |b| {
        b.iter(|| {
            let _mapped = MappedFile::open(black_box(temp_file.path())).unwrap();
        });
    });
    
    group.bench_function("zero_copy_deserializer", |b| {
        let mapped = MappedFile::open(temp_file.path()).unwrap();
        b.iter(|| {
            let _deserializer = ZeroCopyDeserializer::from_mapped_file(black_box(&mapped)).unwrap();
        });
    });
    
    group.bench_function("streaming_deserializer", |b| {
        b.iter(|| {
            let mut streamer = StreamingDeserializer::new(black_box(&test_data), 1024);
            while let Some(_chunk) = streamer.next_chunk() {
                // Process chunk
            }
        });
    });
    
    // Benchmark zero-copy string reading
    let string_data = {
        let mut data = Vec::new();
        let test_str = "Hello, World! This is a test string for zero-copy deserialization.";
        data.extend_from_slice(&(test_str.len() as u32).to_le_bytes());
        data.extend_from_slice(test_str.as_bytes());
        data
    };
    
    group.bench_function("zero_copy_string_read", |b| {
        b.iter(|| {
            let mut deserializer = ZeroCopyDeserializer::new(black_box(&string_data));
            let _str_ref = deserializer.read_string().unwrap();
        });
    });
    
    group.finish();
}

fn bench_large_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_data_structures");
    
    // Create large nested structures
    let large_list = SimpleValue::List((0..10000).map(|i| {
        SimpleValue::List(vec![
            SimpleValue::Integer(i),
            SimpleValue::String(format!("item_{}", i)),
            SimpleValue::Real(i as f64 * 3.14159),
        ])
    }).collect());
    
    let estimated_size = large_list.serialized_size();
    group.throughput(Throughput::Bytes(estimated_size as u64));
    
    group.bench_function("large_nested_list", |b| {
        b.iter(|| {
            let mut buffer = Vec::new();
            large_list.serialize(&mut buffer).unwrap();
            
            let mut cursor = Cursor::new(buffer);
            let _deserialized = SimpleValue::deserialize(&mut cursor).unwrap();
        });
    });
    
    // Test with large session containing many variables
    let mut large_session = SessionState::new();
    for i in 0..5000 {
        large_session.add_variable(
            format!("large_var_{}", i),
            SimpleValue::List(vec![
                SimpleValue::Integer(i),
                SimpleValue::String("x".repeat(100)),
                SimpleValue::Real(i as f64),
            ])
        );
    }
    
    let session_size = large_session.serialized_size();
    group.throughput(Throughput::Bytes(session_size as u64));
    
    group.bench_function("large_session_state", |b| {
        b.iter(|| {
            let mut buffer = Vec::new();
            large_session.serialize(&mut buffer).unwrap();
            
            let mut cursor = Cursor::new(buffer);
            let _deserialized = SessionState::deserialize(&mut cursor).unwrap();
        });
    });
    
    group.finish();
}

fn bench_cross_platform_compatibility(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_platform");
    
    // Test endianness handling
    let test_numbers = vec![
        0u32, 1u32, 255u32, 256u32, 65535u32, 65536u32, u32::MAX,
        0u64, 1u64, u64::MAX,
    ];
    
    group.bench_function("endianness_conversion", |b| {
        b.iter(|| {
            for &num in &test_numbers {
                // Test u32
                let bytes = num.to_le_bytes();
                let _converted = u32::from_le_bytes(bytes);
                
                // Test u64 (cast to u64)
                let num64 = num as u64;
                let bytes64 = num64.to_le_bytes();
                let _converted64 = u64::from_le_bytes(bytes64);
            }
        });
    });
    
    // Test platform-specific optimizations
    use lyra::serialization::platform::{PlatformWriter, PlatformReader};
    
    group.bench_function("platform_writer", |b| {
        b.iter(|| {
            let mut buffer = Vec::new();
            let mut writer = PlatformWriter::new(&mut buffer);
            
            for &num in &test_numbers {
                writer.write_u32(num).unwrap();
                writer.write_u64(num as u64).unwrap();
                writer.write_f64(num as f64).unwrap();
            }
        });
    });
    
    group.bench_function("platform_reader", |b| {
        // Pre-create test data
        let mut buffer = Vec::new();
        {
            let mut writer = PlatformWriter::new(&mut buffer);
            for &num in &test_numbers {
                writer.write_u32(num).unwrap();
                writer.write_u64(num as u64).unwrap();
                writer.write_f64(num as f64).unwrap();
            }
        }
        
        b.iter(|| {
            let cursor = Cursor::new(black_box(&buffer));
            let mut reader = PlatformReader::new(cursor);
            
            for _ in &test_numbers {
                let _u32_val = reader.read_u32().unwrap();
                let _u64_val = reader.read_u64().unwrap();
                let _f64_val = reader.read_f64().unwrap();
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_basic_serialization,
    bench_compression,
    bench_session_persistence,
    bench_zero_copy,
    bench_large_data_structures,
    bench_cross_platform_compatibility
);
criterion_main!(benches);