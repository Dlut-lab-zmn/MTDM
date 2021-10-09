# MTDM
Temporal Knowledge Graph Reasoning Triggered by Memories

To alleviate the time dependence, we propose a memory-triggered decision-making (MTDM) network, which incorporates transient memories, long-short-term memories, and deep memories.
Specifically, a transient learning network considers the latest events as a static knowledge graph, and a time-aware recurrent evolutional network is regarded as a sequence of recurrent evolution units. 
Each evolution unit consists of a structural encoder to aggregate edge information, a time encoder with a gating unit to update attribute representations of entities based on
structural encoder outputs. 
Both the transient learning network and time-aware recurrent evolutional network utilize the crafted residual multi-relational aggregator as the structural encoder to solve the multi-hop coverage problem. 
We also introduce adversarial learning for understanding events dissolution. 
Extensive experiments demonstrate the MTDM alleviates the history dependence and achieves state-of-the-art prediction performance.
Moreover, compared with the most advanced baseline, MTDM performs faster convergence speed and training speed.
