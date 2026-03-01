from network import Network
from classical import dijkstra

print("=== Creating Network ===")
network = Network(num_routers=6)

print("\n=== Initial Active Graph ===")
initial_graph = network.get_active_graph()
for node in initial_graph:
    print(f"{node}: {initial_graph[node]}")

print("\n=== Enter Source and Destination ===")

try:
    start = int(input("Enter Source Router ID: "))
    end = int(input("Enter Destination Router ID: "))

    if start not in initial_graph or end not in initial_graph:
        print("Invalid router ID.")
        exit()

except ValueError:
    print("Please enter valid integers.")
    exit()

print("\n=== Running Multi-Packet Simulation ===")

num_packets = 500
total_delay = 0
packets_delivered = 0
packets_lost = 0

for _ in range(num_packets):

    graph = network.get_active_graph()
    cost, path = dijkstra(graph, start, end)

    if not path:
        packets_lost += 1
        continue

    packet_lost = False
    packet_delay = 0

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]

        link = network.routers[node1].links[node2]

        delay, loss, congestion = link.transmit_packet(packet_size_mb=1)

        if loss:
            packet_lost = True
            break

        packet_delay += delay

    if packet_lost:
        packets_lost += 1
    else:
        packets_delivered += 1
        total_delay += packet_delay

print("\n=== Simulation Results ===")

print("Packets Sent:", num_packets)
print("Packets Delivered:", packets_delivered)
print("Packets Lost:", packets_lost)

if packets_delivered > 0:
    print("Average Delay:", total_delay / packets_delivered)

delivery_ratio = packets_delivered / num_packets
print("Delivery Ratio:", delivery_ratio)

print("\n=== Phase 3 Complete ===")
