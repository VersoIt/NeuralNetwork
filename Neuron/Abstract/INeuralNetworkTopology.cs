namespace NeuralNetwork.Abstract
{
    public interface INeuralNetworkTopology
    {
        public int HiddenLayersCount { get; init; }

        public double LearningRate { get; init; }

        public int InputCount { get; init; }

        public int OutputCount { get; init; }

        public int NeuronsInHiddenLayersCount { get; init; }

        public IReadOnlyList<int> NeuronsInHiddenLayersCounts { get; init; }
    }
}
