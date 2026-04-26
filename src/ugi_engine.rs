use std::process::{Command, Stdio, ChildStdout, ChildStdin, Child};
use std::io::{self, Write, Read};
use std::thread::{self, JoinHandle};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError, RecvTimeoutError};
use std::collections::VecDeque;
use std::time::Duration;

/// Use for running a UGI engine as a child process and communicating with it
pub struct UgiEngine {
    child: Child,
    input_sender: Sender<String>,
    ouput_reciver: Receiver<String>,

    reader_thread: Option<JoinHandle<()>>,
    reader_quit_sender: Sender<bool>,

    writer_thread: Option<JoinHandle<()>>,
    writer_quit_sender: Sender<bool>,

    recived_queue: VecDeque<String>,
}

impl Drop for UgiEngine {
    fn drop(&mut self) {
        // Kill the child process first so the reader's blocking read() unblocks.
        let _ = self.child.kill();
        let _ = self.child.wait();

        // Signal both threads to stop, then join them so they don't spin as orphans.
        let _ = self.reader_quit_sender.send(true);
        let _ = self.writer_quit_sender.send(true);

        if let Some(t) = self.reader_thread.take() { let _ = t.join(); }
        if let Some(t) = self.writer_thread.take() { let _ = t.join(); }
    }
}

impl UgiEngine {
    pub fn new(engine_path: &str) -> UgiEngine {
        let mut engine_process = Command::new(engine_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to start the process");

        let stdout = engine_process.stdout.take().unwrap();
        let stdin = engine_process.stdin.take().unwrap();

        let (input_sender, input_reciver) = mpsc::channel();
        let (ouput_sender, ouput_reciver) = mpsc::channel();

        let (quit_sender_1, quit_reciver_1) = mpsc::channel();
        let (quit_sender_2, quit_reciver_2) = mpsc::channel();

        let reader_thread = thread::spawn(move || {
            let mut reader = UgiReader::new(ouput_sender, quit_reciver_1);
            reader.start(stdout);

        });
        let writer_thread = thread::spawn(move || {
            let mut writer = UgiWriter::new(input_reciver, quit_reciver_2);
            writer.start(stdin);

        });

        return UgiEngine {
            child: engine_process,
            input_sender,
            ouput_reciver,

            reader_thread: Some(reader_thread),
            reader_quit_sender: quit_sender_1,

            writer_thread: Some(writer_thread),
            writer_quit_sender: quit_sender_2,

            recived_queue: VecDeque::new(),

        };

    }

    /// Kills the child process immediately. Safe to call multiple times.
    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    pub fn quit(&mut self) {
        self.send("quit");

        self.reader_quit_sender.send(true).unwrap();
        self.writer_quit_sender.send(true).unwrap();

        self.reader_thread.take().unwrap().join().unwrap();
        self.writer_thread.take().unwrap().join().unwrap();

    }

    pub fn send(&mut self, cmd: &str) {
        self.input_sender.send(cmd.to_string()).expect("Failed to send data to the engine");

    }

    fn try_recive(&mut self) {
        match self.ouput_reciver.try_recv() {
            Ok(s) => {
                self.recived_queue.push_front(s.clone());

            }
            Err(_) => {}

        }

    }

    pub fn recive(&mut self) -> Option<String> {
        self.try_recive();

        if self.recived_queue.len() == 0 {
            return None;

        }

        return self.recived_queue.pop_back();

    }

    /// Block until a line is available from the engine. Returns None if the
    /// reader thread has exited (engine died / channel closed).
    pub fn recive_blocking(&mut self) -> Option<String> {
        if let Some(s) = self.recived_queue.pop_back() {
            return Some(s);
        }
        self.ouput_reciver.recv().ok()
    }

    /// Block until a line is available or the timeout elapses. Returns None
    /// on timeout or if the channel closed.
    pub fn recive_with_timeout(&mut self, timeout: Duration) -> Option<String> {
        if let Some(s) = self.recived_queue.pop_back() {
            return Some(s);
        }
        self.ouput_reciver.recv_timeout(timeout).ok()
    }

}


struct UgiReader {
    data_out: Sender<String>,
    quit_in: Receiver<bool>,

}

impl UgiReader {
    pub fn new(data_out: Sender<String>, quit_in: Receiver<bool>) -> UgiReader {
        return UgiReader {
            data_out,
            quit_in

        };
    
    }

    pub fn start(&mut self, stdout: ChildStdout) {
        let mut stdout_reader: io::BufReader<std::process::ChildStdout> = io::BufReader::new(stdout);
        let mut stdout_buffer: [u8; 4096] = [0; 4096];

        loop {
            match stdout_reader.read(&mut stdout_buffer) {
                Ok(0) => break, // EOF — child process exited; spinning here would burn 100% CPU
                Ok(n) => {
                    let output = String::from_utf8_lossy(&stdout_buffer[..n]).to_string();
                    for l in output.lines() {
                        self.data_out.send(l.to_string()).unwrap();
                    }
                }
                Err(_) => break,
            }

            match self.quit_in.try_recv() {
                Ok(_) | Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {}
            }
        }
    }
    
}

struct UgiWriter {
    data_in: Receiver<String>,
    quit_in: Receiver<bool>,

}

impl UgiWriter {
    pub fn new(data_in: Receiver<String>, quit_in: Receiver<bool>) -> UgiWriter {
        return UgiWriter {
            data_in,
            quit_in

        };
    
    }

    pub fn start(&mut self, mut stdin: ChildStdin) {
        loop {
            // Block up to 100ms waiting for the next command. Timeout lets us
            // periodically check the quit signal without burning CPU.
            match self.data_in.recv_timeout(Duration::from_millis(100)) {
                Ok(s) => {
                    // If the engine process is dead, writing to its stdin fails — just exit.
                    if stdin.write_all(format!("{}\n", s).as_bytes()).is_err() {
                        break;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break, // sender dropped — UgiEngine was dropped
                Err(RecvTimeoutError::Timeout) => {}
            }

            match self.quit_in.try_recv() {
                Ok(_) | Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {}
            }
        }
    }
    
}
