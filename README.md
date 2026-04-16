# CEC: Efficient Cloud-Edge Collaboration Framework for AI Tasks

This is the implementation of **CEC** (Cloud-Edge Collaboration), a framework that leverages vector databases to optimize cloud-edge collaborative AI tasks. 
<!-- CEC optimizes LLM planning results by considering edge resource limitations and task-database semantic similarities, then schedules subtasks across edges and/or the cloud using reinforcement learning. The core scheduling algorithm is referred to as **TPS** (Task Planning and Scheduling) in the code. -->

## Download Data and Pretrained Weights

Large data files and pretrained model weights are hosted on Google Drive. Download them before running:

**[Google Drive: CEC](https://drive.google.com/drive/folders/1bvnJpodxuGuLpeC1Nl5oFIyE_jo12t80)**

| Folder | Contents | Where to place |
|--------|----------|---------------|
| [`data/`](https://drive.google.com/drive/folders/1GWsFp8Z4iNGS35zr8Y8NOIDmUqugeAdN) | `lumos_complex_qa_ground_onetime.jsonl`, `lumos_answer.rar` | `shared_program/program/` and `data/` |
| [`milvus_backup/`](https://drive.google.com/drive/folders/1p8UwZ9hOiO2d4WmAnLrz6prZLQwJ6U-E) | 11 `.npz` files (Milvus collection backup) | `data/milvus_backup/` |
| [`expert_demonstrations/`](https://drive.google.com/drive/folders/1mDpcYEhei0gbJGDr0Mq2cYBrl9l5qCwE) | 5 `.pkl` files (expert demonstration data) | `expert_demonstrations/` |
| [`pretrained_weights/`](https://drive.google.com/drive/folders/1rhy4gSC97z2b9z3ceeCkaHrVO2KraY3w) | `.pth` files for each baseline | `Baselines/<algorithm>/` (match subfolder names) |

## Project Structure

```
source_codes/
├── TPS/                  # Our method (Task Planning and Scheduling algorithm)
│   ├── train_main.py     # Training entry point
│   ├── env_main.py       # RL environment (task scheduling)
│   ├── agent_regular.py  # PPO agent
│   ├── buffer_main.py    # Replay buffer
│   └── utils.py          # Data loading utilities
│
├── Baselines/            # Baseline algorithms
│   ├── TA-subvec/               # TA-0.6: greedy, picks edge with highest vector similarity; falls back to cloud if all < 0.6
│   ├── CE-subvec/               # CE: schedules to lowest-GPU-utilization edge, no vector DB; falls back to cloud when full
│   ├── CO-subvec/               # CO: always schedules to cloud (Cloud-Only)
│   ├── SAC-subvec/
│   ├── DDPG-subvec/
│   ├── DQN-subvec/
│   ├── PPO/
│   ├── BC-subvec/
│   ├── IRL-subvec/
│   ├── GNNPPO-subvec/
│   ├── PPOBC-subvec/
│   ├── RL-subvec-withoutexpert/
│   ├── RL-subvec-withoutpretrain/
│   ├── ppo-vector-subvec/
│   ├── ppo-vectoremotion-subvec/
│   ├── ppo-vectorfield-subvec/
│   └── ppo-vectorkeywords-subvec/
│
├── Agents/               # Agent nodes (deployed on remote servers)
│   ├── agents/           # Edge agent node (lightweight local LLM)
│   └── cloud_agent/      # Cloud agent node (large cloud-hosted LLM)
│
├── shared_program/       # Shared task dispatch & communication module
│   ├── embedding.py      # Text embedding
│   └── program/          # Task dispatch, node communication, vector DB
│
├── expert_demonstrations/          # Expert demonstration data (shared by all algorithms)
│   ├── memory_relu_new_reward.pkl
│   ├── memory_relu_subvec_reward.pkl
│   ├── memory_rl_subvec_reward.pkl
│   ├── memory_sim_new_reward_high_cost.pkl
│   └── memory_sim_subvec_reward_high_cost.pkl
│
└── data/                 # Dataset and vector database setup
    ├── lumos_answer.rar  # Enhanced answers archive
    ├── milvus_backup/    # Exported Milvus collection data (.npz files)
    ├── milvus_scripts/   # Scripts to create and populate Milvus
    │   ├── create_collection.py  # Create Milvus collection schema
    │   ├── insert_result.py      # Insert enhanced answers into Milvus
    │   ├── save_dataset1.py      # Export/restore Milvus data from .npz
    │   └── embedding.py          # Embedding helper for insertion
    └── milvus_init/                      # Vector database initialization scripts (from edge servers)
        └── algorithm_insert_vector/      # Per-node init scripts for both experiment variants
            ├── create_collection_node*.py          # Create collection schema on each edge node
            ├── create_keywords_collection_node*.py # Create keyword vector collection on each edge node
            ├── insert_milvus*_sub.py               # Relevant Cache init: insert agent execution results
            ├── insert_milvus*_insert.py            # Random Cache init: insert randomly sampled ground truth
            ├── insert_keywords_milvus.py           # Insert keyword embeddings alongside Q&A vectors
            ├── get_qa.py                           # Parse task_result_500.txt and Lumos JSONL
            ├── correct.py                          # Fix malformed multi-line answers in task_result.txt
            ├── task_result_500.txt                 # 500 actual agent execution results (Relevant Cache data)
            ├── task_result.txt                     # 3000 pre-computed ground truth answers (Random Cache data)
            └── embedding.py / get_*.py / utils.py  # Embedding and text analysis helpers
```

## System Architecture

The CEC framework consists of three layers:

1. **Algorithm Layer** (TPS / Baselines) - RL-based task planning and scheduling, runs on the training server
2. **Task Dispatch Layer** (shared_program/program) - Sends tasks to agent nodes via TCP, receives results
3. **Agent Execution Layer** (Agents) - Receives tasks, executes them using LLM/SLM (CrewAI framework), returns results

### Agent Nodes

There are two types of agent nodes, reflecting the edge-cloud collaborative setting:

- **`agents/`** (Edge Node) - Deployed on edge servers with a lightweight local LLM (qwen-1.8b). Handles tasks locally with lower latency. Each edge node reports its GPU/CPU resource usage to the training server for scheduling decisions.
- **`cloud_agent/`** (Cloud Node) - Deployed on a cloud server with a larger LLM (qwen14b). Handles tasks that require stronger reasoning capabilities. The cloud node does not report resource status since the scheduler does not need to manage cloud-side load.

Both share the same overall structure (task receiving, CrewAI-based execution, result sending), but differ in LLM model configuration and task processing logic.

### Node Deployment

| Role | Code | Description |
|------|------|-------------|
| **Training Server** | `TPS/` + `shared_program/` | Runs RL training and task dispatch |
| **Edge Node x 3** | `Agents/agents/` | Each runs a local LLM (qwen-1.8b) for task execution |
| **Cloud Node x 1** | `Agents/cloud_agent/` | Runs a large cloud LLM (qwen14b) for complex tasks |

The RL agent's action space has 4 discrete actions, corresponding to 4 execution nodes: 3 edge nodes (action 0/1/2) + 1 cloud node (action 3). In our experiments we used this default 3+1 setup. You can scale the number of edge nodes by deploying `Agents/agents/` on additional servers and updating the IPs in `shared_program/program/paifa_task.py`.

## Dataset and Vector Database

We use the [Lumos](https://github.com/allenai/lumos) complex QA dataset as our task source. Additionally, we review and enhance the provided answers using LLMs (CrewAI framework) and store them in a Milvus vector database. During training, the scheduler retrieves these enhanced answers to compute reward signals by comparing agent outputs against the reference answers via vector similarity.  `lumos_complex_qa_ground_onetime.jsonl`

<!-- ### Setting Up the Vector Database

The vector database must be initialized before training. This corresponds to the two experimental conditions in the paper (Section VI-B):

| Paper Term | Milvus Collection | Init Scripts | Data Source | Description |
|---|---|---|---|---|
| **Relevant Cache** | `crewai_agents1_algorithm1_subvec` | `insert_milvus*_sub.py` | `task_result_500.txt` | 500 actual agent execution results from the Lumos dataset, split across 3 edge nodes. Semantically relevant to incoming tasks. |
| **Random Cache** | `crewai_agents1_algorithm1_insert` | `insert_milvus*_insert.py` | `task_result.txt` + Lumos JSONL | 3000 ground truth answers, only 10 randomly sampled per node. Mostly unrelated to subtasks; tests system adaptability. | -->

**Steps:**

1. **Install and start Milvus** on each edge node (see [Milvus docs](https://milvus.io/docs/install_standalone-docker.md))

2. **Create collections** on each edge node (run on the corresponding node):
   ```bash
   cd data/milvus_init/algorithm_insert_vector
   # Update YOUR_EDGE_NODE*_IP placeholders in the scripts first, then:
   python create_collection_node1.py   # on edge node 1
   python create_collection_node2.py   # on edge node 2
   python create_collection_node3.py   # on edge node 3
   # Also create keyword collections:
   python create_keywords_collection_node1.py
   python create_keywords_collection_node2.py
   python create_keywords_collection_node3.py
   ```

3. **Populate the vector database** (choose one variant):

   - **Relevant Cache** — Insert 500 actual agent execution results (~1/3 per node, deterministic split via `np.random.seed(101)`):
     ```bash
     python insert_milvus1_sub.py   # inserts ~167 entries into node 1
     python insert_milvus2_sub.py   # inserts ~167 entries into node 2
     python insert_milvus3_sub.py   # inserts ~166 entries into node 3
     python insert_keywords_milvus.py  # insert keyword vectors for all nodes
     ```

   - **Random Cache** — Insert 10 randomly sampled ground truth answers per node:
     ```bash
     python insert_milvus1_insert.py
     python insert_milvus2_insert.py
     python insert_milvus3_insert.py
     python insert_keywords_milvus.py
     ```

   - **Option C: Import from backup** — Use the `.npz` files in `data/milvus_backup/` with `data/milvus_scripts/save_dataset1.py` to restore a pre-built collection directly.

## Embedding Model

We use the [M3E-Large](https://huggingface.co/moka-ai/m3e-large) model for text embedding. Deploy it as an OpenAI-compatible API server using Docker:

```bash
docker run -it -d --gpus all \
  --name m3e-large \
  -p 7891:8000 \
  -v /path/to/m3e-large/:/moka-ai/m3e-large \
  gptq_image_v2
```

After starting, update the `base_url` in `shared_program/embedding.py` to point to your deployment address (default: `http://<YOUR_IP>:7891/v1`).

## Prerequisites

- Python 3.8+
- PyTorch
- Gym
- Milvus (vector database)
- CrewAI
- wandb (optional, for logging)
- pymilvus
- scikit-learn

```bash
pip install torch gym pymilvus crewai wandb scikit-learn langchain-community
```

## How to Run

### Step 1: Set Up Vector Database

Follow the instructions in [Dataset and Vector Database](#dataset-and-vector-database) above.

### Step 2: Start Agent Nodes

On **each edge server**, deploy `Agents/agents/` and start:

```bash
cd Agents/agents/

# Receive tasks from training server
python node_receive.py

# Receive forwarded results from other nodes
python node_other_receive.py

# Task execution engine
cd task_process && python main.py

# Send results back to training server
python node_send_host.py

# GPU/CPU resource monitoring (edge nodes only)
cd resource && python resource_update.py
```

On the **cloud server**, deploy `Agents/cloud_agent/` and start the same set of scripts (except `resource_update.py`, which is not needed for the cloud node).

### Step 3: Start Receivers on Training Server

On the training server, start the result receivers:

```bash
cd shared_program/program/

# Receive subtask results from each node (edge1/2/3 + cloud)
python node_receive.py
python node2_receive.py
python node3_receive.py
python node4_receive.py

# Receive final task results
python node_final_receive.py

# Receive subtask execution time from each node
python time_node_receive.py
python time_node2_receive.py
python time_node3_receive.py
python time_node4_receive.py

# Receive final task execution time
python time_node_final_receive.py
```

### Step 4: Start Training

```bash
# TPS (our method)
cd TPS && python train_main.py

# Or any baseline, e.g.:
cd Baselines/SAC-subvec && python train_main.py
```

<!-- ## Baselines

All baselines in `Baselines/` use the **Relevant Cache** (`crewai_agents1_algorithm1_subvec`) vector database initialization by default, matching the primary experimental condition in the paper (Fig. 7). To reproduce the **Random Cache** results (Fig. 8), change `collection_name` in `train_main.py` to `crewai_agents1_algorithm1_insert`.

| Baseline | Paper Name | Description |
|---|---|---|
| `TA-subvec` | TA-0.6 | Greedy: routes subtask to edge with highest vector similarity score; falls back to cloud if all edges score < 0.6 |
| `CE-subvec` | CE | Cloud-Edge without vector DB: round-robins edges by GPU utilization (threshold 70%); falls back to cloud when all edges are full |
| `CO-subvec` | CO | Cloud-Only: always routes all subtasks to cloud LLM |
| `SAC-subvec` | SAC | Off-policy RL (Soft Actor-Critic) with vector state |
| `DDPG-subvec` | DDPG | Off-policy RL (Deep Deterministic Policy Gradient) |
| `DQN-subvec` | DQN | Off-policy RL (Deep Q-Network) |
| `PPO` | PPO | On-policy RL without shared reward or expert pretraining |
| `BC-subvec` | BC | Behavioral Cloning: supervised imitation of expert demonstrations |
| `IRL-subvec` | GAIL | Generative Adversarial Imitation Learning |
| `GNNPPO-subvec` | GAT-PPO | PPO with Graph Attention Network for subtask dependency modeling |
| `PPOBC-subvec` | BC-PPO | PPO initialized via BC then fine-tuned online |
| `RL-subvec-withoutexpert` | TPS-no-expert | TPS ablation: online training only, no expert demonstrations |
| `RL-subvec-withoutpretrain` | TPS-no-pretrain | TPS ablation: no pre-training phase |
| `ppo-vector-subvec` | TPS v | TPS with content similarity only (no field/sentiment/keyword) |
| `ppo-vectorfield-subvec` | TPS v+f | TPS with content + field similarity |
| `ppo-vectoremotion-subvec` | TPS v+e | TPS with content + sentiment similarity |
| `ppo-vectorkeywords-subvec` | TPS v+w | TPS with content + keyword similarity | -->

## Configuration

Before running, update the following IP addresses and ports in the code to match your deployment environment:

| File | What to change |
|------|---------------|
| `shared_program/program/node_receive.py` (and node2/3/4) | `server.bind(('YOUR_IP', port))` |
| `shared_program/program/time_node_receive.py` (and 2/3/4) | `server.bind(('YOUR_IP', port))` |
| `shared_program/program/paifa_task.py` | Agent node IPs in `send_file()` calls |
| `shared_program/program/get_tasks.py` | Agent node IPs and resource file paths |
| `Agents/agents/node_receive.py` | `server.bind(('YOUR_IP', port))` |
| `Agents/agents/task_process/main.py` | Milvus connection address |
| `Agents/agents/task_process/agents.py` | LLM API endpoint and model name |
| `Agents/cloud_agent/node_receive.py` | `server.bind(('YOUR_IP', port))` |
| `Agents/cloud_agent/task_process/agents.py` | Cloud LLM API endpoint and model name |
| `shared_program/embedding.py` | Embedding model API address (`base_url`) |
| `data/milvus_scripts/*.py` | Milvus connection address |
