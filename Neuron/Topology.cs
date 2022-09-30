using NeuralNetwork.Abstract;

namespace NeuralNetwork
{
    public struct NeuralNetworkTopology : INeuralNetworkTopology
    {

        public NeuralNetworkTopology(int inputCount, int outputCount, double learningRate, params int[] neuronsInHiddenLayersCounts)
        {
            InputCount = inputCount;
            OutputCount = outputCount;

            NeuronsInHiddenLayersCounts = new int[neuronsInHiddenLayersCounts.Length];
            HiddenLayersCount = neuronsInHiddenLayersCounts.Length;
            LearningRate = learningRate;

            neuronsInHiddenLayersCounts.CopyTo((Array)NeuronsInHiddenLayersCounts, 0);
            NeuronsInHiddenLayersCount = neuronsInHiddenLayersCounts.Sum();
        }

        public int HiddenLayersCount { get; init; }

        public int NeuronsInHiddenLayersCount { get; init; }

        public double LearningRate { get; init; }

        public int InputCount { get; init; }

        public int OutputCount { get; init; }

        public IReadOnlyList<int> NeuronsInHiddenLayersCounts { get; init; }
    }
}
