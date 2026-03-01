import random
from network import Network

print("=== Creating Network ===")
network = Network(num_routers=6)

print("\n=== Active Topology ===")
graph = network.get_active_graph()

for node in graph:
    print(f"Router {node}: {graph[node]}")

print("\n=== Random Link Traffic Test ===")

if network.links:

    for i in range(5):
        link = random.choice(network.links)
        delay, loss, congestion = link.transmit_packet(packet_size_mb=1)

        print(
            f"Packet {i+1} | "
            f"Link {link.node1}-{link.node2} | "
            f"Delay={delay} | "
            f"Loss={loss} | "
            f"Congestion={congestion}"
        )

    print("\n=== Full Network Evaluation ===")

    for link in network.links:
        delay, loss, congestion = link.transmit_packet(packet_size_mb=1)

        print(
            f"Link {link.node1}-{link.node2} | "
            f"Delay={delay} | "
            f"Loss={loss} | "
            f"Congestion={congestion}"
        )

else:
    print("No links available.")

print("\n=== Phase 1 Complete ===")
