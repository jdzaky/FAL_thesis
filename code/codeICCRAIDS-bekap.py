#!/usr/bin/env python3
"""
SDN Controller Dataset - FIXED VERSION
- Fair AHP scoring (normalized efficiency, removed FLOOD bias)
- Dynamic IP allocation for all scenarios
- Improved resource monitoring and timeout handling
- FIXED: KeyError 'nodes' for tree topologies
- ADDED: Retry for OpenDaylight on RESOURCE_CAPTURE_FAILED
"""
import subprocess, csv, time, os, logging, json, re, sys, random
import numpy as np, psutil, socket, requests
from datetime import datetime
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
import traceback
from scipy import stats
import requests
import threading

CONTROLLERS = {
    'ryu': {'image': 'osrg/ryu:latest', 'port': 6633, 'startup_wait': 8},
    'floodlight': {'image': 'latarc/floodlight:latest', 'port': 6653, 'startup_wait': 10},
    'onos': {'image': 'latarc/onos:latest', 'port': 6653, 'startup_wait': 45}
}

SCENARIOS = [
    # STAR topologies (simple, centralized)
#    {'topo': 'star', 'nodes': 8, 'bw': '30M', 'name': 'Simple_Star_8nodes_30M', 'description': 'Baseline star'},
#    {'topo': 'star', 'nodes': 4, 'bw': '10M', 'name': 'Minimal_Star_4nodes_10M', 'description': 'Minimal star'},
#    # TREE topologies (hierarchical, multi-level)
#    {'topo': 'tree', 'depth': 2, 'fanout': 3, 'bw': '50M', 'name': 'Tree_Depth2_Fanout3_50M', 'description': 'Hierarchical tree'},
#    {'topo': 'tree', 'depth': 3, 'fanout': 2, 'bw': '80M', 'name': 'Tree_Depth3_Fanout2_80M', 'description': 'Deep tree'},
#    # LINEAR topologies (sequential chain)
#    {'topo': 'linear', 'nodes': 12, 'bw': '50M', 'name': 'Linear_12nodes_50M', 'description': 'Chain topology'},
#    {'topo': 'linear', 'nodes': 8, 'bw': '100M', 'name': 'Linear_8nodes_100M', 'description': 'High-bandwidth chain'},
#    # MESH topologies (highly connected)
    {'topo': 'mesh', 'nodes': 6, 'bw': '100M', 'name': 'Mesh_6nodes_100M', 'description': 'Fully connected mesh'},
    {'topo': 'mesh', 'nodes': 8, 'bw': '5M', 'name': 'Mesh_8nodes_5M', 'description': 'Mesh with low bandwidth'},
    # RING topologies (circular)
    {'topo': 'ring', 'nodes': 10, 'bw': '60M', 'name': 'Ring_10nodes_60M', 'description': 'Ring topology'},
    {'topo': 'ring', 'nodes': 16, 'bw': '80M', 'name': 'Ring_16nodes_80M', 'description': 'Large ring'},
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'generation_ahp.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

CSV_FILE = os.path.join(DATA_DIR, 'sdn_dataset_ahp.csv')
CSV_HEADER = [
    'sample_id', 'iteration', 'timestamp', 'controller_name', 'scenario_name',
    'nodes', 'topology_type', 'bandwidth_demand', 'throughput_mbps', 'latency_ms',
    'packet_loss_percent', 'score_reliability', 'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
    'flow_setup_time_ms', 'score_compliance', 'score_efficiency', 'score_stability',
    'ahp_score', 'best_controller_label', 'score_gap', 'resource_method','connectivity_status'
]

def get_scenario_node_count(scenario):
    """Calculate number of hosts based on topology type."""
    topo_type = scenario.get('topo', 'star')
    if topo_type == 'star':
        return scenario.get('nodes', 8)
    elif topo_type == 'linear':
        return scenario.get('nodes', 8)
    elif topo_type == 'mesh':
        return scenario.get('nodes', 6)
    elif topo_type == 'ring':
        return scenario.get('nodes', 8)
    elif topo_type == 'tree':
        depth = scenario.get('depth', 2)
        fanout = scenario.get('fanout', 2)
        leaves = fanout ** (depth - 1)
        return leaves
    else:
        return 8  # fallback

def parse_memory_string(mem_str):
    try:
        match = re.search(r'([\d.]+)\s*([KMG]i?)B', mem_str, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            unit = match.group(2).upper()
            if unit in ['G', 'GI']:
                return val * 1024
            elif unit in ['M', 'MI']:
                return val
            elif unit in ['K', 'KI']:
                return val / 1024
    except Exception as e:
        logging.debug(f"Memory parse error: {e}")
    return 0.0

def check_docker_daemon():
    try:
        result = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logging.info(f"Docker daemon running (version: {result.stdout.strip()})")
            return True
    except Exception as e:
        logging.error(f"Docker check failed: {str(e)}")
        return False

def is_container_running(container_name):
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and container_name in result.stdout
    except:
        return False

def get_docker_stats_hybrid(container_name, ctrl_name):
    """Docker stats + psutil Java process fallback with debug logging."""
    try:
        result = subprocess.run(
            ['docker', 'stats', '--no-stream', '--format', '{{json .}}', container_name],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            stats = json.loads(result.stdout.strip())
            cpu_str = stats.get('CPUPerc', '0%').strip().rstrip('%')
            cpu = float(cpu_str) if cpu_str else 0.0
            mem_mb = parse_memory_string(stats.get('MemUsage', '0B'))
            
            # PERBAIKAN: Terima sampel selama data valid (CPU/MEM tidak keduanya nol)
            if cpu > 0 or mem_mb > 0:
                logging.debug(f"[{ctrl_name}] DOCKER_STATS: CPU={cpu:.2f}%, MEM={mem_mb:.2f}MB")
                return cpu, mem_mb, True
                
            logging.debug(f"[{ctrl_name}] DOCKER_STATS: Low resources (CPU={cpu}, MEM={mem_mb}), trying psutil...")
    except Exception as e:
        logging.debug(f"[{ctrl_name}] DOCKER_STATS_ERROR: {e}")

    # Psutil fallback: detect ANY Java process
    logging.debug(f"[{ctrl_name}] PSUTIL: Scanning all Java processes...")
    try:
        java_procs = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'java' in proc.name().lower():
                    java_procs.append(proc)
                    logging.debug(f"[{ctrl_name}] PSUTIL: Found Java PID {proc.pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Use first Java process if available
        if java_procs:
            try:
                proc = java_procs[0]
                cpu = proc.cpu_percent(interval=0.1)
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                
                # PERBAIKAN: Terima sampel selama salah satu resource terukur
                if cpu > 0 or mem_mb > 0:
                    logging.debug(f"[{ctrl_name}] PSUTIL_FALLBACK: Using PID {proc.pid}, CPU={cpu:.2f}%, MEM={mem_mb:.2f}MB")
                    return cpu, mem_mb, True
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logging.debug(f"[{ctrl_name}] PSUTIL_FALLBACK_ERROR: {e}")
    except Exception as e:
        logging.debug(f"[{ctrl_name}] PSUTIL_SCAN_ERROR: {e}")

    logging.debug(f"[{ctrl_name}] Resource capture: no valid samples available")
    return 0.0, 0.0, False  # Tetap kembalikan False jika benar-benar gagal

def setup_network_learning_switch(net, ctrl_name, host_port):
    logging.info(f"[{ctrl_name}] Setup with learning switch rules...")
    for switch in net.switches:
        try:
            switch.cmd(f'ovs-vsctl set bridge {switch.name} protocols=OpenFlow13')
            switch.cmd(f'ovs-vsctl set-controller {switch.name} tcp:127.0.0.1:{host_port}')
            switch.cmd(f'ovs-vsctl set-fail-mode {switch.name} standalone')
            time.sleep(0.2)
        except Exception as e:
            logging.error(f"[{ctrl_name}] Switch config error: {e}")
            return False
    logging.info(f"[{ctrl_name}] Setup complete (standalone mode)")
    return True

def setup_network_onos_special(net, ctrl_name, host_port):
    logging.info(f"[{ctrl_name}] Setup with ONOS fwd app activation...")
    for switch in net.switches:
        try:
            switch.cmd(f'ovs-vsctl set bridge {switch.name} protocols=OpenFlow13')
            switch.cmd(f'ovs-vsctl set-controller {switch.name} tcp:127.0.0.1:{host_port}')
            switch.cmd(f'ovs-vsctl set-fail-mode {switch.name} secure')
            time.sleep(0.2)
        except Exception as e:
            logging.error(f"[{ctrl_name}] Switch config error: {e}")
            return False
    logging.info(f"[{ctrl_name}] Waiting 25s for ONOS startup...")
    time.sleep(25)
    logging.info(f"[{ctrl_name}] Activating fwd app...")
    for retry in range(5):
        try:
            resp = requests.post(
                "http://localhost:8181/onos/v1/applications/org.onosproject.fwd/active",
                auth=("onos", "rocks"), timeout=5
            )
            if resp.status_code in (200, 204):
                logging.info(f"[{ctrl_name}] fwd app activated")
                time.sleep(5)
                return True
        except requests.exceptions.RequestException as e:
            logging.debug(f"[{ctrl_name}] REST attempt {retry+1} failed: {e}")
            time.sleep(2)
    logging.warning(f"[{ctrl_name}] fwd app activation failed, continuing anyway")
    return True

def test_connectivity_with_bootstrap(h1, h2, ctrl_name, max_attempts=120, net=None):
    logging.info(f"[{ctrl_name}] Testing connectivity with ARP bootstrap...")
    h1.cmd(f'arp -d {h2.IP()} 2>/dev/null')
    h2.cmd(f'arp -d {h1.IP()} 2>/dev/null')
    h1.popen(f'arping -c 5 -w 2 {h2.IP()}', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    h2.popen(f'arping -c 5 -w 2 {h1.IP()}', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    for attempt in range(max_attempts):
        try:
            timeout = 3
            result = h1.popen(
                f'ping -c 1 -W {timeout} {h2.IP()}',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = result.communicate(timeout=timeout + 1)
            if result.returncode == 0:
                logging.info(f"[{ctrl_name}] Connectivity OK at attempt {attempt+1}")
                return True
            else:
                # Log detail error ping
                logging.debug(f"[{ctrl_name}] Ping attempt {attempt+1} failed: returncode={result.returncode}, stderr='{stderr.strip()}'")
        except Exception as e:
            logging.debug(f"[{ctrl_name}] Attempt {attempt+1} exception: {e}")

        # Re-trigger ARP setiap 30 attempt
        if attempt == 30 or attempt == 60:
            logging.debug(f"[{ctrl_name}] Re-triggering ARP at attempt {attempt+1}")
            h1.popen(f'arping -c 3 -w 1 {h2.IP()}', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
        else:
            time.sleep(0.5)

    # === DEBUG AKHIR: CEK FLOW & STATUS SWITCH ===
    logging.error(f"[{ctrl_name}] No connectivity after {max_attempts} attempts")
    logging.debug(f"[{ctrl_name}] DEBUG: Checking OVS flows and switch status...")

    try:
        # Cek flow di semua switch
        net = h1.net  # Mininet network object
        for switch in net.switches:
            flows = switch.cmd('ovs-ofctl dump-flows ' + switch.name)
            logging.debug(f"[{ctrl_name}] Flows on {switch.name}:\n{flows}")
        
        # Cek koneksi controller
        for switch in net.switches:
            controllers = switch.cmd('ovs-vsctl get-controller ' + switch.name)
            logging.debug(f"[{ctrl_name}] Controller for {switch.name}: {controllers.strip()}")
            
            bridge_status = switch.cmd('ovs-vsctl show')
            logging.debug(f"[{ctrl_name}] Bridge status:\n{bridge_status}")
    except Exception as e:
        logging.debug(f"[{ctrl_name}] Failed to collect OVS debug info: {e}")

    return False

class RobustMeasurement:
    def __init__(self, src, dst, ctrl_name):
        self.src = src
        self.dst = dst
        self.dst_ip = dst.IP()
        self.ctrl_name = ctrl_name

    def _cleanup(self):
        try:
            self.src.popen('pkill -9 iperf3 ping arping 2>/dev/null').wait()
            self.dst.popen('pkill -9 iperf3 2>/dev/null').wait()
        except:
            pass
        time.sleep(0.3)

    def ping_test(self, count=15):
        try:
            result = self.src.popen(
                f'ping -c {count} -W 2 {self.dst_ip}',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, _ = result.communicate(timeout=count+5)
            match = re.search(r'min/avg/max.*=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', stdout)
            if match:
                return {
                    'min': float(match.group(1)), 'avg': float(match.group(2)),
                    'max': float(match.group(3)), 'jitter': float(match.group(4))
                }
        except Exception as e:
            logging.debug(f"[{self.ctrl_name}] Ping error: {e}")
        return None

    def tcp_test(self, duration=10, port=5001):
        self._cleanup()
        time.sleep(0.5)
        try:
            srv = self.dst.popen(f'iperf3 -s -p {port}', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
            cli = self.src.popen(f'timeout {duration+2} iperf3 -c {self.dst_ip} -p {port} -t {duration} -J',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, _ = cli.communicate(timeout=duration+3)
            if '{' in stdout:
                try:
                    data = json.loads(stdout)
                    bps = data['end']['sum_sent']['bits_per_second']
                    mbps = bps / 1e6
                    logging.debug(f"[{self.ctrl_name}] TCP: {mbps:.2f} Mbps")
                    return round(mbps, 2)
                except Exception as e:
                    logging.debug(f"[{self.ctrl_name}] TCP JSON error: {e}")
            srv.terminate()
            srv.wait(timeout=1)
        except Exception as e:
            logging.debug(f"[{self.ctrl_name}] TCP error: {e}")
        finally:
            self._cleanup()
        return 0.0

    def udp_test(self, duration=10, bandwidth='10M', port=None):
        if port is None:
            port = random.randint(5200, 5900)
        self._cleanup()
        time.sleep(0.5)
        try:
            srv = self.dst.popen(f'iperf3 -s -p {port} -1', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            cli = self.src.popen(
                f'timeout {duration+5} iperf3 -c {self.dst_ip} -p {port} -u -b {bandwidth} -t {duration} -J',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, _ = cli.communicate(timeout=duration+6)
            if '{' in stdout:
                try:
                    data = json.loads(stdout)
                    total = data['end']['sum']['packets']
                    lost = data['end']['sum']['lost_packets']
                    pct = (lost / total) * 100 if total > 0 else 0
                    logging.debug(f"[{self.ctrl_name}] UDP: {pct:.3f}% loss")
                    return round(pct, 3)
                except Exception as e:
                    logging.debug(f"[{self.ctrl_name}] UDP JSON error: {e}")
            srv.terminate()
            srv.wait(timeout=1)
        except Exception as e:
            logging.debug(f"[{self.ctrl_name}] UDP error: {e}")
        finally:
            self._cleanup()
        return 0.0

class AHPCalculator:
    def __init__(self):
        self.weights = {
            'main': {'compliance': 0.724, 'efficiency': 0.193, 'stability': 0.083},
            'compliance': {'throughput': 0.090, 'latency': 0.354, 'reliability': 0.556},
            'efficiency': {'cpu': 0.751, 'memory': 0.249},
            'stability': {'jitter': 0.750, 'flow_setup': 0.250}
        }

    def calculate_score(self, metrics, requirements):
        comp = self._compliance_score(metrics, requirements)
        eff = self._efficiency_score(metrics)
        stab = self._stability_score(metrics)
        rel = self._reliability_score(metrics, requirements)
        total = (
            self.weights['main']['compliance'] * comp +
            self.weights['main']['efficiency'] * eff +
            self.weights['main']['stability'] * stab
        )
        return {
            'total': round(total, 4),
            'compliance': round(comp, 4),
            'efficiency': round(eff, 4),
            'stability': round(stab, 4),
            'reliability': round(rel, 4)
        }

    def _reliability_score(self, metrics, requirements):
        return min((100 - metrics.get('packet_loss_percent', 100)) / max(requirements.get('reliability_req', 99.5), 1), 1.0)

    def _compliance_score(self, metrics, requirements):
        t = min(metrics.get('throughput_mbps', 0) / max(requirements.get('throughput_req', 50), 1), 1.0)
        l = min(1.0, max(0, 1.0 - (metrics.get('latency_ms', 999) - requirements.get('latency_req', 50)) / max(requirements.get('latency_req', 50), 1)))
        r = min((100 - metrics.get('packet_loss_percent', 100)) / max(requirements.get('reliability_req', 99.5), 1), 1.0)
        w = self.weights['compliance']
        return w['throughput'] * t + w['latency'] * l + w['reliability'] * r

    def _efficiency_score(self, metrics):
        cpu_raw = metrics.get('cpu_usage_percent', 50)
        mem_raw = metrics.get('memory_usage_mb', 300)
        cpu_norm = max(0, 1.0 - (cpu_raw / 100.0))
        mem_norm = max(0, 1.0 - np.log10(mem_raw + 1) / np.log10(2000))
        w = self.weights['efficiency']
        return w['cpu'] * cpu_norm + w['memory'] * mem_norm

    def _stability_score(self, metrics):
        jitter = max(0, 1.0 - metrics.get('jitter_ms', 50) / 50.0)
        flow = max(0, 1.0 - metrics.get('flow_setup_time_ms', 200) / 300.0)
        w = self.weights['stability']
        return w['jitter'] * jitter + w['flow_setup'] * flow

class StarTopo(Topo):
    def __init__(self, num_hosts=8):
        Topo.__init__(self)
        s1 = self.addSwitch('s1')
        for i in range(1, num_hosts + 1):
            h = self.addHost(f'h{i}')
            self.addLink(h, s1)

class TreeTopo(Topo):
    def __init__(self, depth=2, fanout=2):
        Topo.__init__(self)
        self.depth = depth
        self.fanout = fanout
        self.switch_count = 0
        root = self.addSwitch(f's{self._get_switch_id()}')
        self._add_tree_level(root, depth, fanout)

    def _get_switch_id(self):
        self.switch_count += 1
        return self.switch_count

    def _add_tree_level(self, parent_switch, depth, fanout):
        if depth <= 0:
            return
        for i in range(fanout):
            if depth == 1:
                h = self.addHost(f'h{parent_switch}-{i}')
                self.addLink(h, parent_switch)
            else:
                child_switch = self.addSwitch(f's{self._get_switch_id()}')
                self.addLink(child_switch, parent_switch)
                self._add_tree_level(child_switch, depth - 1, fanout)

class LinearTopo(Topo):
    def __init__(self, num_nodes=8):
        Topo.__init__(self)
        switches = []
        for i in range(1, (num_nodes // 2) + 1):
            s = self.addSwitch(f's{i}')
            switches.append(s)
            if i > 1:
                self.addLink(switches[i-2], s)
        host_count = 1
        for switch in switches:
            for j in range(2):
                h = self.addHost(f'h{host_count}')
                self.addLink(h, switch)
                host_count += 1

class MeshTopo(Topo):
    def __init__(self, num_nodes=6):
        Topo.__init__(self)
        switches = []
        for i in range(num_nodes):
            s = self.addSwitch(f's{i+1}')
            switches.append(s)
        for i in range(len(switches)):
            for j in range(i+1, len(switches)):
                self.addLink(switches[i], switches[j])
        for i, switch in enumerate(switches):
            h = self.addHost(f'h{i+1}')
            self.addLink(h, switch)

class RingTopo(Topo):
    def __init__(self, num_nodes=8):
        Topo.__init__(self)
        switches = []
        for i in range(num_nodes):
            s = self.addSwitch(f's{i+1}')
            switches.append(s)
            h = self.addHost(f'h{i+1}')
            self.addLink(h, s)
        for i in range(num_nodes):
            next_switch = switches[(i + 1) % num_nodes]
            self.addLink(switches[i], next_switch)

def get_topology_class(scenario):
    topo_type = scenario.get('topo', 'star')
    if topo_type == 'star':
        nodes = scenario.get('nodes', 8)
        return StarTopo(num_hosts=nodes)
    elif topo_type == 'tree':
        depth = scenario.get('depth', 2)
        fanout = scenario.get('fanout', 2)
        return TreeTopo(depth=depth, fanout=fanout)
    elif topo_type == 'linear':
        nodes = scenario.get('nodes', 8)
        return LinearTopo(num_nodes=nodes)
    elif topo_type == 'mesh':
        nodes = scenario.get('nodes', 6)
        return MeshTopo(num_nodes=nodes)
    elif topo_type == 'ring':
        nodes = scenario.get('nodes', 8)
        return RingTopo(num_nodes=nodes)
    else:
        return StarTopo(num_hosts=8)

class CustomRemoteController(RemoteController):
    def checkListening(self):
        return True

def full_cleanup():
    logging.info("System cleanup...")
    subprocess.run(['sudo', 'pkill', '-9', '-f', 'mininet'], capture_output=True, timeout=2)
    subprocess.run(['sudo', 'mn', '-c'], capture_output=True, timeout=3)
    try:
        cids = subprocess.run(['docker', 'ps', '-aq'], capture_output=True, text=True, timeout=5)
        if cids.returncode == 0 and cids.stdout.strip():
            for cid in cids.stdout.strip().split('\n'):
                if cid:
                    subprocess.run(['docker', 'kill', cid], capture_output=True, timeout=1)
                    subprocess.run(['docker', 'rm', '-f', cid], capture_output=True, timeout=1)
    except:
        pass
    time.sleep(1)

def collect_resource_samples_hybrid(container_name, ctrl_name, duration_sec, interval=0.5):
    samples = []
    end_time = time.time() + duration_sec
    sample_count = 0
    logging.info(f"[{ctrl_name}] RESOURCE_MONITOR: Starting {duration_sec}s collection (interval={interval}s)")
    while time.time() < end_time:
        cpu, mem, success = get_docker_stats_hybrid(container_name, ctrl_name)
        if success:
            samples.append((datetime.now(), cpu, mem))
            sample_count += 1
            logging.debug(f"[{ctrl_name}] SAMPLE_{sample_count}: CPU={cpu:.2f}%, MEM={mem:.2f}MB")
        else:
            logging.debug(f"[{ctrl_name}] SAMPLE_{sample_count+1}: FAILED")
        time.sleep(interval)
    logging.info(f"[{ctrl_name}] RESOURCE_MONITOR: Collected {len(samples)} samples out of ~{int(duration_sec/interval)}")
    return samples

def measure_performance_with_mitigation(net, traffic_info, container_name, ctrl_name, nodes):
    metrics = {
        'throughput_mbps': 0.0, 'latency_ms': 0.0, 'packet_loss_percent': 0.0,
        'jitter_ms': 0.0, 'flow_setup_time_ms': 0.0, 'cpu_usage_percent': 0.0,
        'memory_usage_mb': 0.0, 'resource_method': 'NONE', 'connectivity_status': 'FAILED'
    }
    if len(net.hosts) < 2:
        return metrics
    h1, h2 = net.hosts[0], net.hosts[-1]
    h1.setIP(f'10.0.0.1/24')
    h2.setIP(f'10.0.0.{min(200, 2 + nodes)}/24')
    time.sleep(1)
    if ctrl_name in ['opendaylight', 'onos']:
        logging.info(f"[{ctrl_name}] Controller mode active (skip data plane test)")
        metrics['connectivity_status'] = 'CONTROLLER'
    else:
        h1.net = net
        h2.net = net
        if not test_connectivity_with_bootstrap(h1, h2, ctrl_name, max_attempts=60):
            logging.warning(f"[{ctrl_name}] Controller attempt FAILED - switching to standalone")
            for switch in net.switches:
                try:
                    switch.cmd(f'ovs-vsctl set-fail-mode {switch.name} standalone')
                except:
                    pass
            time.sleep(3)
            metrics['connectivity_status'] = 'STANDALONE'
            if not test_connectivity_with_bootstrap(h1, h2, ctrl_name, max_attempts=30, net=net):
                logging.error(f"[{ctrl_name}] Final connectivity check FAILED")
                return metrics
        else:
            metrics['connectivity_status'] = 'CONTROLLER'

    time.sleep(3)
    flow_setup_time = 25 + (nodes - 1) * 2.5
    m = RobustMeasurement(h1, h2, ctrl_name)
    perf_metrics = {}
    test_duration = 40

    def run_tests():
        nonlocal perf_metrics
        try:
            ping_data = m.ping_test(count=15)
            if ping_data and 'avg' in ping_data:
                perf_metrics['latency_ms'] = round(ping_data['avg'], 3)
                perf_metrics['jitter_ms'] = round(ping_data['jitter'], 3)
                logging.info(f"[{ctrl_name}] âœ… Ping test completed: avg latency={perf_metrics['latency_ms']} ms, jitter={perf_metrics['jitter_ms']} ms")
            else:
                logging.warning(f"[{ctrl_name}] âŒ Ping test failed or returned no data")

            time.sleep(2)
            tcp = m.tcp_test(duration=10, port=5001)
            perf_metrics['throughput_mbps'] = tcp if tcp else 0.0
            logging.info(f"[{ctrl_name}] âœ… TCP test completed: throughput={perf_metrics['throughput_mbps']:.2f} Mbps")

            time.sleep(2)
            udp = m.udp_test(duration=10, bandwidth=traffic_info['bw'])
            perf_metrics['packet_loss_percent'] = udp if udp is not None else 0.0
            logging.info(f"[{ctrl_name}] âœ… UDP test completed: packet loss={perf_metrics['packet_loss_percent']:.3f}%")
        except Exception as e:
            logging.error(f"[{ctrl_name}] Test error: {e}", exc_info=True)

    test_thread = threading.Thread(target=run_tests)
    test_thread.start()
    logging.info(f"[{ctrl_name}] Resource monitoring ({test_duration}s) - HYBRID...")
    resource_samples = collect_resource_samples_hybrid(container_name, ctrl_name, test_duration, interval=0.5)
    test_thread.join(timeout=test_duration + 5)
    metrics.update({
        'throughput_mbps': perf_metrics.get('throughput_mbps', 0.0),
        'latency_ms': perf_metrics.get('latency_ms', 0.0),
        'jitter_ms': perf_metrics.get('jitter_ms', 0.0),
        'packet_loss_percent': perf_metrics.get('packet_loss_percent', 0.0),
        'flow_setup_time_ms': round(flow_setup_time, 2),
        'cpu_usage_percent': 0.0,
        'memory_usage_mb': 0.0,
        'resource_method': 'NONE'
    })
    if resource_samples:
        cpus = [s[1] for s in resource_samples]
        mems = [s[2] for s in resource_samples]
        metrics['cpu_usage_percent'] = round(np.percentile(cpus, 95), 2)
        metrics['memory_usage_mb'] = round(np.percentile(mems, 95), 2)
        metrics['resource_method'] = 'hybrid_p95'
        logging.info(f"[{ctrl_name}] FINAL_RESOURCES: CPU={metrics['cpu_usage_percent']}% (P95), MEM={metrics['memory_usage_mb']}MB (P95), Samples={len(resource_samples)}")
    else:
        metrics['resource_method'] = 'no_samples'
        logging.warning(f"[{ctrl_name}] FINAL_RESOURCES: No samples collected - setting CPU/MEM to 0.0")
    return metrics

def install_l2switch_feature(container_name, ctrl_name, max_retries=20):
    """
    Install L2Switch via karaf CLI (lebih reliable daripada REST).
    """
    logging.info(f"[{ctrl_name}] Installing L2Switch via Karaf CLI...")
    
    for attempt in range(max_retries):
        try:
            # Cek apakah sudah installed
            check_cmd = [
                'docker', 'exec', container_name, 'sh', '-c',
                'echo "feature:list | grep l2switch" | /opt/karaf/bin/client'
            ]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
            
            if 'odl-l2switch-switch' in check_result.stdout and 'Started' in check_result.stdout:
                logging.info(f"[{ctrl_name}] âœ… L2Switch already installed")
                time.sleep(5)
                return True
            
            # Install
            install_cmd = [
                'docker', 'exec', container_name, 'sh', '-c',
                'echo "feature:install odl-l2switch-switch" | /opt/karaf/bin/client'
            ]
            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=15)
            
            logging.debug(f"[{ctrl_name}] Install attempt {attempt+1}: {result.stdout[:150]}")
            
            if result.returncode == 0 or 'exception' not in result.stdout.lower():
                logging.info(f"[{ctrl_name}] âœ… L2Switch installed via Karaf")
                time.sleep(15)
                return True
                
        except Exception as e:
            logging.debug(f"[{ctrl_name}] Karaf attempt {attempt+1}: {e}")
        
        time.sleep(5)
    
    logging.warning(f"[{ctrl_name}] L2Switch install failed - controller running with default forwarding")
    return False


def wait_for_odl_ready(container_name, ctrl_name, timeout=180):
    """
    Tunggu ODL siap dengan cek:
    1. Karaf started di logs
    2. REST endpoint http://localhost:8181 respond
    3. Port 6653 listening
    """
    logging.info(f"[{ctrl_name}] Waiting for ODL full startup...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Cek Karaf log
        try:
            result = subprocess.run(['docker', 'logs', container_name],
                                  capture_output=True, text=True, timeout=10)
            if "Karaf started" not in result.stdout:
                logging.debug(f"[{ctrl_name}] Karaf not started yet")
                time.sleep(5)
                continue
        except Exception as e:
            logging.debug(f"[{ctrl_name}] Log check error: {e}")
            time.sleep(5)
            continue
        
        # Cek REST endpoint (8181)
        try:
            resp = requests.get("http://localhost:8181/restconf/operational/", 
                              auth=("admin", "admin"), timeout=5)
            if resp.status_code in [200, 204, 401]:  # 401 = auth issue tapi endpoint live
                logging.info(f"[{ctrl_name}] REST endpoint responsive")
                break
        except requests.exceptions.RequestException:
            logging.debug(f"[{ctrl_name}] REST not ready yet")
            time.sleep(5)
            continue
    
    # Cek port 6653
    time.sleep(10)
    port_check = _is_port_open_in_container(container_name, 6653)
    logging.info(f"[{ctrl_name}] Port 6653: {'OPEN' if port_check else 'CLOSED (OK, akan open saat switch connect)'}")
    logging.info(f"[{ctrl_name}] âœ… ODL Ready")
    return True

def setup_network_odl_controller_mode(net, ctrl_name, host_port):
    """
    Setup switches dengan CONTROLLER mode (fail-mode=secure).
    Jangan standalone - harus proper controller communication.
    """
    logging.info(f"[{ctrl_name}] Configuring switches (CONTROLLER mode - secure fail-mode)...")
    
    for switch in net.switches:
        try:
            switch.cmd(f'ovs-vsctl set bridge {switch.name} protocols=OpenFlow13')
            switch.cmd(f'ovs-vsctl set-controller {switch.name} tcp:127.0.0.1:{host_port}')
            # PENTING: secure fail-mode agar switch tunggu controller
            switch.cmd(f'ovs-vsctl set-fail-mode {switch.name} secure')
            switch.cmd(f'ovs-vsctl set-controller {switch.name} tcp:127.0.0.1:{host_port}')
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"[{ctrl_name}] Switch config error: {e}")
            return False
    
    logging.info(f"[{ctrl_name}] Switches in CONTROLLER mode (fail-mode=secure)")
    time.sleep(8)
    
    # Cek controller connection
    success_count = 0
    for switch in net.switches:
        try:
            status = switch.cmd('ovs-vsctl get-controller ' + switch.name)
            if '127.0.0.1' in status:
                success_count += 1
                logging.debug(f"[{ctrl_name}] {switch.name}: Controller connected")
            else:
                logging.warning(f"[{ctrl_name}] {switch.name}: Controller not connected yet")
        except Exception as e:
            logging.error(f"[{ctrl_name}] Status check error: {e}")
    
    if success_count > 0:
        logging.info(f"[{ctrl_name}] {success_count}/{len(net.switches)} switches connected to controller")
        return True
    else:
        logging.error(f"[{ctrl_name}] No switches connected to controller")
        return False

# --- MAIN GENERATION FUNCTION WITH RETRY ---
MAX_ODL_RETRIES = 2

def generate_sample(ctrl_name, ctrl_process, host_port, scenario, sample_id, all_samples, ctrl_info):
    node_count = get_scenario_node_count(scenario)
    logging.info(f"\n[Sample {sample_id}] {ctrl_name} + {scenario['name']} ({node_count} nodes)")
    
    net = None
    retry_count = 0
    while retry_count <= (MAX_ODL_RETRIES if ctrl_name == 'opendaylight' else 0):
        try:
            topo = get_topology_class(scenario)
            net = Mininet(
                topo=topo,
                controller=lambda name: CustomRemoteController(name, ip='127.0.0.1', port=host_port),
                link=TCLink, switch=OVSSwitch, autoSetMacs=True, waitConnected=False
            )
            logging.info(f"[{ctrl_name}] Starting Mininet...")
            net.start()
            time.sleep(5)

            if ctrl_name == 'opendaylight':
                # Setup controller mode
                if not setup_network_odl_controller_mode(net, ctrl_name, host_port):
                    raise Exception("ODL controller mode setup failed")
                
                # Install L2Switch via Karaf CLI
                container_name = ctrl_process[1] if ctrl_process and ctrl_process[0] == 'docker' else None
                install_l2switch_feature(container_name, ctrl_name, max_retries=20)
                time.sleep(10)
                
            elif ctrl_name == 'onos':
                if not setup_network_onos_special(net, ctrl_name, host_port):
                    raise Exception("ONOS setup failed")
            else:
                if not setup_network_learning_switch(net, ctrl_name, host_port):
                    raise Exception("Learning switch setup failed")

            traffic_info = {'bw': scenario['bw']}
            container_name = ctrl_process[1] if ctrl_process and ctrl_process[0] == 'docker' else None
            perf = measure_performance_with_mitigation(net, traffic_info, container_name, ctrl_name, node_count)

            if perf.get('connectivity_status') == 'FAILED':
                logging.warning(f"[Sample {sample_id}] Discarding invalid sample")
                return False

            net.stop()
            time.sleep(2)
            full_cleanup()
            time.sleep(5)

            requirements = {
                'throughput_req': int(scenario['bw'].rstrip('M')) * 0.8,
                'latency_req': node_count * 1.5 + 20,
                'reliability_req': 99.0 + (node_count * 0.1)
            }
            ahp_calc = AHPCalculator()
            ahp_result = ahp_calc.calculate_score(perf, requirements)

            row = {
                'sample_id': sample_id, 'iteration': 0, 'timestamp': datetime.now().isoformat(),
                'controller_name': ctrl_name, 'scenario_name': scenario['name'],
                'nodes': node_count, 'topology_type': scenario['topo'],
                'bandwidth_demand': scenario['bw'],
                'throughput_mbps': perf['throughput_mbps'], 'latency_ms': perf['latency_ms'],
                'packet_loss_percent': perf['packet_loss_percent'], 'score_reliability': ahp_result['reliability'],
                'jitter_ms': perf['jitter_ms'],
                'cpu_usage_percent': perf['cpu_usage_percent'], 'memory_usage_mb': perf['memory_usage_mb'],
                'flow_setup_time_ms': perf['flow_setup_time_ms'],
                'score_compliance': ahp_result['compliance'], 'score_efficiency': ahp_result['efficiency'],
                'score_stability': ahp_result['stability'],
                'ahp_score': ahp_result['total'], 'best_controller_label': '', 'score_gap': 0,
                'resource_method': perf.get('resource_method', 'NONE'),
                'connectivity_status': perf.get('connectivity_status', 'UNKNOWN')
            }
            all_samples.append(row)
            logging.info(f"[Sample {sample_id}] âœ… SUCCESS")
            return True

        except KeyboardInterrupt:
            logging.info("Interrupted")
            if net:
                net.stop()
            full_cleanup()
            sys.exit(0)
        except Exception as e:
            logging.error(f"[Sample {sample_id}] âŒ FAILED: {str(e)}")
            try:
                if net:
                    net.stop()
            except:
                pass
            full_cleanup()
            if ctrl_name == 'opendaylight' and retry_count < MAX_ODL_RETRIES:
                retry_count += 1
                logging.info(f"[{ctrl_name}] Retrying ({retry_count}/{MAX_ODL_RETRIES})...")
                time.sleep(15)
                continue
            return False

    return False

def assign_best_controller_labels(all_samples):
    scenarios = {}
    for sample in all_samples:
        key = sample['scenario_name']
        if key not in scenarios:
            scenarios[key] = []
        scenarios[key].append(sample)
    for scenario_samples in scenarios.values():
        if len(scenario_samples) < 2:
            continue
        best_sample = max(scenario_samples, key=lambda x: x['ahp_score'])
        best_ctrl = best_sample['controller_name']
        scores = sorted([s['ahp_score'] for s in scenario_samples], reverse=True)
        score_gap = scores[0] - scores[1] if len(scores) > 1 else 0
        for sample in scenario_samples:
            sample['best_controller_label'] = best_ctrl
            sample['score_gap'] = round(score_gap, 4)

def check_container_health(container_name, ctrl_name, max_wait=60):
    logging.info(f"[{ctrl_name}] Health check...")
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            result = subprocess.run(
                ['docker', 'inspect', container_name, '--format={{.State.Status}}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                time.sleep(1)
                continue
            status = result.stdout.strip()
            if status == 'running':
                return True
            elif status == 'exited':
                logging.error(f"[{ctrl_name}] Container exited")
                return False
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            logging.debug(f"[{ctrl_name}] Health check error: {e}")
        time.sleep(2)
    logging.error(f"[{ctrl_name}] Health check timeout")
    return False

# =============== HELPER UNTUK OPENDAYLIGHT ROBUST ===============
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _wait_for_odl_log_ready(container_name, timeout=200):
    """
    Wait until 'Karaf started' appears in container logs.
    This is the most reliable indicator that ODL is fully initialized.
    """
    logging.info(f"[opendaylight] Waiting for 'Karaf started' in logs (timeout={timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ['docker', 'logs', container_name],
                capture_output=True, text=True, timeout=10
            )
            if "Karaf started" in result.stdout:
                logging.info("[opendaylight] âœ… Karaf started detected in logs")
                return True
        except Exception as e:
            logging.debug(f"[opendaylight] Log read error: {e}")
        time.sleep(3)
    logging.error("[opendaylight] âŒ Timeout waiting for Karaf startup")
    return False

def _is_java_process_alive(container_name):
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "pgrep", "-f", "java"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except:
        return False

def _is_port_open_in_container(container_name, port):
    try:
        hex_port = format(port, '04X')
        cmd = f"cat /proc/net/tcp* 2>/dev/null | awk '{{print $2}}' | cut -d: -f2 | grep -q {hex_port}"
        result = subprocess.run(
            ["docker", "exec", container_name, "sh", "-c", cmd],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except:
        return False

def _dump_container_logs(container_name):
    try:
        result = subprocess.run(['docker', 'logs', container_name], capture_output=True, text=True, timeout=15)
        log_tail = '\n'.join(result.stdout.splitlines()[-30:])  # ambil 30 baris terakhir
        logging.error(f"[{container_name}] --- Container logs (last 30 lines) ---\n{log_tail}")
    except Exception as e:
        logging.error(f"[{container_name}] Failed to fetch logs: {e}")

def start_controller_docker(ctrl_name, ctrl_info):
    container = f"{ctrl_name}-batch"
    logging.info(f"[{ctrl_name}] Starting Docker container...")
    try:
        subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=10)
        time.sleep(1)

        memory = '4g' if ctrl_name in ['opendaylight', 'onos'] else '2g'
        cpus = '2.5' if ctrl_name in ['opendaylight', 'onos'] else '2'

        if ctrl_name == 'opendaylight':
            cmd = [
                'docker', 'run', '-d', '--name', container,
                '--network', 'host',
                '--memory', memory, '--cpus', cpus,
                '-e', 'JAVA_OPTS=-Xms1g -Xmx2g -XX:+UseG1GC',
                ctrl_info['image']
            ]
        elif ctrl_name == 'ryu':
            cmd = [
                'docker', 'run', '-d', '--name', container,
                '--network', 'host',
                '--memory', memory, '--cpus', cpus,
                ctrl_info['image'],
                'ryu-manager', '--verbose', 'ryu.app.simple_switch_13'
            ]
        else:
            cmd = [
                'docker', 'run', '-d', '--name', container,
                '--network', 'host',
                '--memory', memory, '--cpus', cpus,
                ctrl_info['image']
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logging.error(f"[{ctrl_name}] Docker run failed: {result.stderr}")
            return False, None, None

        container_id = result.stdout.strip()[:12]
        logging.info(f"[{ctrl_name}] Container: {container_id}")

        if ctrl_name == 'opendaylight':
            if not wait_for_odl_ready(container, ctrl_name, timeout=180):
                logging.error(f"[{ctrl_name}] Startup failed")
                subprocess.run(['docker', 'rm', '-f', container], timeout=10)
                return False, None, None
            return True, ('docker', container), ctrl_info['port']
        else:
            time.sleep(ctrl_info['startup_wait'])
            if not check_container_health(container, ctrl_name, max_wait=30):
                logging.error(f"[{ctrl_name}] Health check failed")
                subprocess.run(['docker', 'rm', '-f', container], timeout=10)
                return False, None, None
            return True, ('docker', container), ctrl_info['port']

    except Exception as e:
        logging.error(f"[{ctrl_name}] Startup error: {str(e)}")
        subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=10)
        return False, None, None



def main():
    logging.info("="*80)
    logging.info("SDN CONTROLLER DATASET GENERATION - FIXED VERSION")
    logging.info("="*80)
    if not check_docker_daemon():
        logging.error("Docker daemon required")
        sys.exit(1)
    all_samples = []
    sample_id = 1
    try:
        for iteration, scenario in enumerate(SCENARIOS):
            logging.info(f"\nSCENARIO {iteration + 1}/{len(SCENARIOS)}: {scenario['name']}")
            for ctrl_name, ctrl_info in CONTROLLERS.items():
                logging.info(f"\n[{ctrl_name}] Starting...")
                success, ctrl_process, port = start_controller_docker(ctrl_name, ctrl_info)
                if not success:
                    logging.warning(f"[{ctrl_name}] Failed to start")
                    continue
                success = generate_sample(ctrl_name, ctrl_process, port, scenario, sample_id, all_samples, ctrl_info)
                if success:
                    sample_id += 1
                full_cleanup()
                time.sleep(5)
        if all_samples:
            assign_best_controller_labels(all_samples)
            logging.info(f"\nWriting {len(all_samples)} samples to CSV...")
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()
                writer.writerows(all_samples)
            logging.info(f"Dataset saved: {CSV_FILE}")
        else:
            logging.error("No samples generated")
    except KeyboardInterrupt:
        logging.info("Interrupted")
        if all_samples:
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()
                writer.writerows(all_samples)
            logging.info(f"Partial dataset saved: {CSV_FILE}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
