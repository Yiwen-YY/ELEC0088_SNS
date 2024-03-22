import socket
import threading

def handle_client(client_socket, address):
    print(f"[+] Accepted connection from {address}")

    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        print(f"Received data from {address}: {data.decode()}")

        client_socket.send(data)

    print(f"[-] Connection from {address} closed")
    client_socket.close()

def main():
    host = "127.0.0.1"
    port = 8888

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"[*] Listening on {host}:{port}")

    while True:
        client_socket, address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(client_socket, address))
        client_thread.start()

if __name__ == "__main__":
    main()
