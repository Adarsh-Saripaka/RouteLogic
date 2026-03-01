import random


class Link:
    def __init__(self, node1, node2, cost,
                 bandwidth_mbps,
                 base_delay_ms,
                 congestion_prob,
                 failure_prob):

        self.node1 = node1
        self.node2 = node2
        self.cost = cost
        self.bandwidth_mbps = bandwidth_mbps
        self.base_delay_ms = base_delay_ms
        self.congestion_prob = congestion_prob
        self.failure_prob = failure_prob
        self.is_active = True
        self.queue_load_mb = 0

    def simulate_link_state(self):
        self.is_active = random.random() > self.failure_prob
        congestion = random.random() < self.congestion_prob
        return congestion

    def transmit_packet(self, packet_size_mb=1):
        if not self.is_active:
            return None, True, False

        congestion = self.simulate_link_state()

        transmission_delay = (packet_size_mb / self.bandwidth_mbps) * 1000
        queue_delay = self.queue_load_mb * 0.5
        total_delay = self.base_delay_ms + transmission_delay + queue_delay

        if congestion:
            total_delay *= random.uniform(1.5, 2.5)
            self.queue_load_mb += packet_size_mb

        packet_loss = False
        if congestion and random.random() < 0.25:
            packet_loss = True

        self.queue_load_mb = max(0, self.queue_load_mb - 0.5)

        return total_delay, packet_loss, congestion


class Router:
    def __init__(self, router_id):
        self.id = router_id
        self.links = {}

    def add_link(self, neighbor, link):
        self.links[neighbor.id] = link


class Network:
    def __init__(self, num_routers=8):
        self.routers = {}
        self.links = []
        self.num_routers = num_routers
        self.create_routers()
        self.create_topology()

    def create_routers(self):
        for i in range(self.num_routers):
            self.routers[i] = Router(i)

    def create_topology(self):
        for i in range(self.num_routers):
            for j in range(i + 1, self.num_routers):
                if random.random() < 0.5:

                    cost = random.randint(1, 10)
                    bandwidth = random.randint(50, 200)
                    base_delay = random.uniform(1, 10)
                    congestion_prob = random.uniform(0.1, 0.3)
                    failure_prob = random.uniform(0.01, 0.05)

                    link = Link(
                        i, j, cost,
                        bandwidth,
                        base_delay,
                        congestion_prob,
                        failure_prob
                    )

                    self.links.append(link)
                    self.routers[i].add_link(self.routers[j], link)
                    self.routers[j].add_link(self.routers[i], link)

    def get_active_graph(self):
        graph = {router_id: {} for router_id in self.routers}

        for link in self.links:
            link.simulate_link_state()
            if link.is_active:
                graph[link.node1][link.node2] = link.cost
                graph[link.node2][link.node1] = link.cost

        return graph
