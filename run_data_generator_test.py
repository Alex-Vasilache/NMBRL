import subprocess
import time
import os
import threading
import datetime
import pickle
import shutil
import portalocker


def stream_watcher(identifier, stream):
    for line in stream:
        print(f"[{identifier}] {line}", end="")
    stream.close()


def concurrent_reader(stop_event, buffer_path):
    """Periodically tries to read the buffer file to test concurrency."""
    while not stop_event.wait(timeout=7):  # Check every 7 seconds
        print("[READER] Attempting to read buffer...")
        if not os.path.exists(buffer_path):
            print("[READER] Buffer file does not exist yet. Waiting.")
            continue

        try:
            with open(buffer_path, "rb") as f:
                # Use portalocker for a shared lock, will wait if exclusive lock is held
                portalocker.lock(f, portalocker.LOCK_SH)

                all_data = []
                while True:
                    try:
                        chunk = pickle.load(f)
                        all_data.extend(chunk)
                    except EOFError:
                        break  # End of file

                print(
                    f"[READER] Successfully read buffer. Records so far: {len(all_data)}"
                )
                portalocker.unlock(f)
        except portalocker.LockException:
            print("[READER] Could not acquire lock, writer is busy. Will try again.")
        except Exception as e:
            print(f"[READER] Error reading buffer: {e}")


def run_test():
    """
    Creates a unique folder, starts the data generator with it, waits 60s,
    stops it, reads the buffer, and cleans up.
    """
    # 1. Create unique folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"temp/{timestamp}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"--- Created unique folder for this run: {save_folder} ---")

    stop_file_path = os.path.join(save_folder, "stop_signal.tmp")
    buffer_path = os.path.join(save_folder, "buffer.pkl")

    # Start concurrent reader
    reader_stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=concurrent_reader, args=(reader_stop_event, buffer_path)
    )
    reader_thread.start()

    # 2. Start data generator
    command = [
        "python",
        "-u",
        "learning/dynamic_data_generator.py",
        "--save-folder",
        save_folder,
    ]
    print("--- Starting data generator test ---")

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    stdout_thread = threading.Thread(
        target=stream_watcher, args=("STDOUT", proc.stdout)
    )
    stderr_thread = threading.Thread(
        target=stream_watcher, args=("STDERR", proc.stderr)
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        # 3. Wait for 1 minute
        print("Data generator process started. Letting it run for 60 seconds...")
        time.sleep(60)

    finally:
        # 4. Signal stop
        print(
            f"--- Stopping data generator by creating stop file: {stop_file_path} ---"
        )
        with open(stop_file_path, "w") as f:
            pass

        # 5. Wait for termination
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print("--- Data generator did not stop gracefully. Terminating. ---")
            proc.terminate()

        stdout_thread.join()
        stderr_thread.join()
        print("--- Data generator process stopped ---")

        # Stop the reader thread
        print("--- Stopping concurrent reader ---")
        reader_stop_event.set()
        reader_thread.join()

        # 6. Read the buffer file
        print(f"--- Performing final read of buffer file: {buffer_path} ---")
        if os.path.exists(buffer_path):
            all_data = []
            try:
                with open(buffer_path, "rb") as f:
                    while True:
                        try:
                            chunk = pickle.load(f)
                            all_data.extend(chunk)
                        except EOFError:
                            break
                print(
                    f"Successfully read buffer. Total records collected: {len(all_data)}"
                )
                if all_data:
                    print(
                        f"First record (input shape, output shape): {all_data[0][0].shape}, {all_data[0][1].shape}"
                    )
            except Exception as e:
                print(f"Error reading buffer file: {e}")
        else:
            print("Buffer file not found.")

        # 7. Clean up
        print(f"--- Cleaning up test folder: {save_folder} ---")
        shutil.rmtree(save_folder, ignore_errors=True)

        print("--- Data generator test finished ---")


if __name__ == "__main__":
    run_test()
