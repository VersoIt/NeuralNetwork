namespace NeuralNetwork.Abstract
{
    public interface INeuralNetwork
    {
        public INeuralLayer this[int index]
        {
            get => Layers[index];
            set => Layers[index] = value;
        }

        public INeuralNetworkTopology Topology { get; }

        public IList<INeuralLayer> Layers { get; }

        public Neuron FeedForward(IReadOnlyList<double> inputSignals);

        public double Learn(IReadOnlyList<double> expected, double[,] inputs, int epoch);
    }
}
