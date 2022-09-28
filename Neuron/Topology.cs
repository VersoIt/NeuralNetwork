
namespace Neuron
{
    public struct Topology
    {

        public Topology(int inputCount, int outputCount, double learningRate, params int[] neuronsInHiddenLayersCounts)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            NeuronsInHiddenLayersCounts = new int[neuronsInHiddenLayersCounts.Length];
            LearningRate = learningRate;

            neuronsInHiddenLayersCounts.CopyTo(NeuronsInHiddenLayersCounts, 0);
        }

        public double LearningRate { get; }

        public int InputCount { get; }

        public int OutputCount { get; }

        public int[] NeuronsInHiddenLayersCounts { get; }
    }
}
