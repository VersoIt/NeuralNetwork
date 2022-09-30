namespace NeuralNetwork.Abstract
{
    public interface INeuralLayer
    {
        public Neuron[] Neurons { get; }

        public Neuron this[int index]
        {
            get => Neurons[index];
            set => Neurons[index] = value;
        }

        public IReadOnlyList<double> GetSignals();

        public void FeedForwardNeurons(IReadOnlyList<double> previousLayerSignals);

        public void LearnNeurons(INeuralLayer previousLayer, double learningRate);
    }
}
