namespace Neuron
{
    public class NeuralNetwork
    {

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Layer this[int index]
        {
            get => Layers[index];
            set => Layers[index] = value;
        }
        
        public Topology Topology { get; }

        public List<Layer> Layers { get; }

        private void CreateInputLayer()
        {
            Neuron[] inputNeurons = new Neuron[Topology.InputCount];
            for (int index = 0; index < Topology.InputCount; ++index)
            {
                var neuron = new DefaultNeuron(1, NeuronTypes.Input);
                inputNeurons[index] = neuron;
            }

            var inputLayer = new Layer(inputNeurons, NeuronTypes.Input);
            Layers.Add(inputLayer);
        }

        private void CreateOutputLayer()
        {
            Neuron[] outputNeurons = new Neuron[Topology.InputCount];
            var lastLayer = Layers.Last();

            for (int index = 0; index < Topology.OutputCount; ++index)
            {
                var neuron = new DefaultNeuron(lastLayer.Neurons.Count(), NeuronTypes.Output);
                outputNeurons[index] = neuron;
            }

            var outputLayer = new Layer(outputNeurons, NeuronTypes.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        {
            foreach (var neuronsCount in Topology.NeuronsInHiddenLayersCounts)
            {
                Neuron[] hiddenNeurons = new Neuron[Topology.InputCount];
                var lastLayer = Layers.Last();

                for (int index = 0; index < neuronsCount; ++index)
                {
                    var neuron = new DefaultNeuron(lastLayer.Neurons.Count(), NeuronTypes.Input);
                    hiddenNeurons[index] = neuron;
                }

                var inputLayer = new Layer(hiddenNeurons, NeuronTypes.Input);
                Layers.Add(inputLayer);
            }
        }

        public Neuron FeedForward(List<double> inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
                return Layers.Last().Neurons.First();
            else
                return Layers.Last().Neurons.OrderByDescending(x => x.Output).First();
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int index = 1; index < Layers.Count; ++index)
            {
                var previousLayerSignals = Layers[index - 1].GetSignals();
                var layer = Layers[index];

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            if (Topology.InputCount != inputSignals.Count)
                throw new Exception("Количество входных сигналов должно быть равно количеству нейронов перовно слоя...");

            for (int index = 0; index < inputSignals.Count; ++index)
            {
                var signal = new double[inputSignals.Count];
                var neuron = Layers.First().Neurons[index];

                neuron.FeedForward(signal);
            }
        }
    }
}
