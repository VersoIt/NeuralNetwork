using System.Collections;
using System.Reflection.Emit;
using NeuralNetwork.Abstract;


namespace NeuralNetwork
{
    public class NeuralLayer : IEnumerable, INeuralLayer
    {
        public const int NeuronsInOneThread = 1024;

        public NeuralLayer(Neuron[] neurons)
        {
            Neurons = neurons;
        }

        public Neuron[] Neurons { get; }

        public Neuron this[int index]
        {
            get => Neurons[index];
            set => Neurons[index] = value;
        }

        public IReadOnlyList<double> GetSignals()
        {
            var signals = new double[Neurons.Count()];

            for (int index = 0; index < signals.Length; ++index)
                signals[index] = Neurons[index].Output;

            return signals;
        }

        private bool optimizationFlow() => Neurons.Length >= NeuronsInOneThread * 1.5;

        public void FeedForwardNeurons(IReadOnlyList<double> previousLayerSignals)
        {
            if (optimizationFlow())
            {
                FeedForwardNeuronsByThreads(previousLayerSignals);
                return;
            }

            foreach (var neuron in Neurons)
            {
                neuron.FeedForward(previousLayerSignals);
            }
        }

        public void LearnNeurons(INeuralLayer previousLayer, double learningRate)
        {
            if (optimizationFlow())
            {
                LearnNeuronsByThreads(previousLayer, learningRate);
                return;
            }

            for (int pos = 0; pos < Neurons.Count(); ++pos)
            {
                var neuron = Neurons[pos];
                for (int k = 0; k < previousLayer.Neurons.Count(); ++k)
                {
                    var previousNeuron = previousLayer.Neurons[k];
                    var error = previousNeuron.Weights[pos] * previousNeuron.Delta;

                    neuron.Learn(error, learningRate);
                }
            }
        }

        private void LearnNeuronsByThreads(INeuralLayer previousLayer, double learningRate)
        {
            List<Thread> threads = new List<Thread>();
            int subThreadsCount = Neurons.Count() / NeuronsInOneThread;

            for (int part = 0; part < subThreadsCount; ++part)
            {
                var thread = new Thread(() =>
                {
                    int neurounsCount = Neurons.Count();
                    int finish = NeuronsInOneThread * (part + 1);
                    for (int pos = part * NeuronsInOneThread; pos < (finish > neurounsCount ? neurounsCount : finish); ++pos)
                    {
                        var neuron = this[pos];
                        for (int k = 0; k < previousLayer.Neurons.Count(); ++k)
                        {
                            var previousNeuron = previousLayer.Neurons[k];
                            var error = previousNeuron.Weights[pos] * previousNeuron.Delta;

                            neuron.Learn(error, learningRate);
                        }
                    }
                });

                threads.Add(thread);
                thread.Start();
            }

            for (int pos = Neurons.Count() - Neurons.Count() % NeuronsInOneThread; pos < Neurons.Count(); ++pos)
            {
                var neuron = this[pos];
                for (int k = 0; k < previousLayer.Neurons.Count(); ++k)
                {
                    var previousNeuron = previousLayer.Neurons[k];
                    var error = previousNeuron.Weights[pos] * previousNeuron.Delta;

                    neuron.Learn(error, learningRate);
                }
            }

            foreach (var thread in threads)
                thread.Join();
        }

        private void FeedForwardNeuronsByThreads(IReadOnlyList<double> previousLayerSignals)
        {
            List<Thread> threads = new List<Thread>();
            int subThreadsCount = Neurons.Count() / NeuronsInOneThread;

            for (int part = 0; part < subThreadsCount; ++part)
            {
                var thread = new Thread(() =>
                {
                    int neurounsCount = Neurons.Count();
                    int finish = NeuronsInOneThread * (part + 1);
                    for (int pos = part * NeuronsInOneThread; pos < (finish > neurounsCount ? neurounsCount : finish); ++pos)
                    {
                        this[pos].FeedForward(previousLayerSignals);
                    }
                });

                threads.Add(thread);
                thread.Start();
            }

            for (int i = Neurons.Count() - Neurons.Count() % NeuronsInOneThread; i < Neurons.Count(); ++i)
            {
                Neurons[i].FeedForward(previousLayerSignals);
            }

            foreach (var thread in threads)
                thread.Join();
        }

        public override string ToString() => $"{Neurons.First().NeuronType} layer";

        public IEnumerator GetEnumerator() => Neurons.GetEnumerator();
    }
}
