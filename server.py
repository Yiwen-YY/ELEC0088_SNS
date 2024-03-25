import socket
import threading
import pandas as pd

def handle_client(client_socket, address):
    print(f"[+] Accepted connection from {address}")

    tavg_forecast = pd.read_csv("tavg_forecast.csv")
    
    while True:
        data = client_socket.recv(1024).decode()

        if not data:
            break
        print(f"Received data from {address}: {data}")

        if "hello" in data:
            response = "Hello! I'm the Oracle, your smart robot. :D"
        elif "today" in data:
            response = "Today is 2021-12-31."
        elif "temperature" in data:
            if "tomorrow" in data:
                tavg = tavg_forecast[tavg_forecast['date'] == '2022-01-01']['tavg'].values[0]
                response = f"The average temperature of tomorrow is {tavg}"
        else:
            response = "Sorry, I can't understand ;_;"

        client_socket.send(response.encode())

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
