import socket
import threading
import pandas as pd
from datetime import date, timedelta
import re

def handle_client(client_socket, address):
    print(f"[+] Accepted connection from {address}")

    tmin_forecast = pd.read_csv("tmin_forecast.csv")
    tmax_forecast = pd.read_csv("tmax_forecast.csv")
    
    while True:
        data = client_socket.recv(1024).decode()
        today = date.today()

        pattern = r'\d{4}-\d{1,2}-\d{1,2}'
        match = re.search(pattern, data)

        if not data:
            break
        print(f"Received data from {address}: {data}")

        if "hello" in data:
            response = "Hello! I'm the Oracle, your smart robot. :D"
        elif "today" in data:
            response = f"Today is {today}."
        elif "temperature" in data:
            if "tomorrow" in data:
                tomorrow = today + timedelta(days=1)
                tomorrow = tomorrow.strftime("%Y-%m-%d")
                tmin = tmin_forecast[tmin_forecast['date'] == tomorrow]['tmin'].values[0]
                tmax = tmax_forecast[tmax_forecast['date'] == tomorrow]['tmax'].values[0]
                response = "The minimum temperature of tomorrow is " + "{:.1f}".format(tmin) + ", the maximum temperature of tomorrow is " + "{:.1f}".format(tmax) + "."
            if match:
                date_str = match.group()
                tmin = tmin_forecast[tmin_forecast['date'] == date_str]['tmin'].values[0]
                tmax = tmax_forecast[tmax_forecast['date'] == date_str]['tmax'].values[0]
                response = "The minimum temperature of " + date_str + " is " + "{:.1f}".format(tmin) + ", the maximum temperature of " + date_str + " is " + "{:.1f}".format(tmax) + "."
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
