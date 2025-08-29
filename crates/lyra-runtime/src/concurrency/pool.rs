use lyra_core::value::Value;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::OnceLock;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// Simple global thread pool for parallel primitives
struct ThreadPool {
    tx: Sender<Box<dyn FnOnce() + Send + 'static>>,
}

static POOL: OnceLock<ThreadPool> = OnceLock::new();

fn thread_pool() -> &'static ThreadPool {
    POOL.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Box<dyn FnOnce() + Send + 'static>>();
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let shared_rx = Arc::new(Mutex::new(rx));
        for _ in 0..workers {
            let rx_cl = shared_rx.clone();
            thread::spawn(move || loop {
                let job_opt = {
                    let lock = rx_cl.lock().unwrap();
                    lock.recv()
                };
                match job_opt {
                    Ok(job) => job(),
                    Err(_) => break,
                }
            });
        }
        ThreadPool { tx }
    })
}

pub(crate) fn spawn_task<F>(f: F) -> Receiver<Value>
where
    F: FnOnce() -> Value + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Value>();
    let job = Box::new(move || {
        let _ = tx.send(f());
    }) as Box<dyn FnOnce() + Send + 'static>;
    let _ = thread_pool().tx.send(job);
    rx
}

#[derive(Debug)]
pub(crate) struct ThreadLimiter {
    max: usize,
    in_use: Mutex<usize>,
    cv: Condvar,
}

impl ThreadLimiter {
    pub(crate) fn new(max: usize) -> Self {
        Self { max, in_use: Mutex::new(0), cv: Condvar::new() }
    }
    pub(crate) fn acquire(&self) {
        let mut guard = self.in_use.lock().unwrap();
        while *guard >= self.max {
            guard = self.cv.wait(guard).unwrap();
        }
        *guard += 1;
    }
    pub(crate) fn release(&self) {
        let mut guard = self.in_use.lock().unwrap();
        if *guard > 0 {
            *guard -= 1;
        }
        self.cv.notify_one();
    }
    pub(crate) fn max_permits(&self) -> usize { self.max }
}
