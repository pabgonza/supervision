"""
Simple socket client to test line crossing events from ROI demo.

Usage:
    python socket_client_test.py [--port 7777] [--host localhost]
"""

import argparse
import socket


def main():
    parser = argparse.ArgumentParser(
        description="Test client for ROI demo line crossing events"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7777, help="Server port (default: 7777)"
    )

    args = parser.parse_args()

    print(f"Connecting to {args.host}:{args.port}...")

    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((args.host, args.port))
        print("Connected! Waiting for line crossing events...\n")
        print("Format: timestamp,in_increment,out_increment")
        print("-" * 60)

        buffer = ""
        while True:
            data = client.recv(1024).decode("utf-8")
            if not data:
                print("\nConnection closed by server")
                break

            buffer += data
            lines = buffer.split("\n")
            buffer = lines[-1]  # Keep incomplete line in buffer

            for line in lines[:-1]:
                if line.strip():
                    try:
                        timestamp, in_inc, out_inc = line.split(",")
                        print(
                            f"{timestamp} | IN: +{in_inc.strip()} | OUT: +{out_inc.strip()}"
                        )
                    except ValueError:
                        print(f"Invalid data: {line}")

    except KeyboardInterrupt:
        print("\n\nDisconnected by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
