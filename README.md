# LST-RE
Long-short Term Recurrent Evolutional Network
We proposed a long short-term recurrent evolutional network (LST-RE), which incorporates the static entity representation, the deep memory, the short term entity attribute representation, and the long term entity attribute representation. Specifically, a short-term memory network considers the latest events as a static knowledge graph, and a long-term memory network is regarded as a sequence of recurrent short-term memory networks. Each recurrent unit contains a structural encoder and a time encoder. We introduce a residual multi-relational aggregator as the structural encoder to solve the multi-hop coverage problem. Extensive experiments demonstrate the LST-RE alleviates the history dependence problem and realizes the state-of-the-art performance. Moreover, compared with the most advanced baseline,  LST-RE achieves up to 2 times the training acceleration even combines the long-short term networks.