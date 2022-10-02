using System.Data;
using NeuralNetwork.Tools;
using NeuralNetwork.Abstract;


namespace NeuralNetwork
{
    public class FeedForwardNeuralNetwork : INeuralNetwork
    {
        public FeedForwardNeuralNetwork(INeuralNetworkTopology topology)
        {
            Topology = topology;
            Layers = new List<INeuralLayer>(topology.HiddenLayersCount + 2);

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public INeuralNetworkTopology Topology { get; }

        public IList<INeuralLayer> Layers { get; }

        public INeuralLayer this[int index]
        {
            get => Layers[index];
            set => Layers[index] = value;
        }

        private void Normalize(double[,] inputs)
        {
            for (int column = 0; column < inputs.GetLength(1); ++column)
            {

                var sum = 0.0;
                // Среднее значение сигналов нейрона
                for (int row = 0; row < inputs.GetLength(0); ++row)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);

                // Стандартное квадратичное отклонение нейрона
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); ++row)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                var standardError = Math.Sqrt(error / inputs.GetLength(0));

                // Новое значение сигнала
                for (int row = 0; row < inputs.GetLength(0); ++row)
                {
                    inputs[row, column] = (inputs[row, column] - average) / standardError;
                }
            }
        }

        private double[,] Scale(double[,] inputs)
        {
            double[,] result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int column = 0; column < inputs.GetLength(1); ++column)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for (int row = 1; row < inputs.GetLength(0); ++row)
                {
                    var item = inputs[row, column];

                    max = Math.Max(item, max);
                    min = Math.Min(item, min);
                }

                var divider = max - min;
                for (int row = 0; row < inputs.GetLength(0); ++row)
                {
                    result[row, column] = (inputs[row, column] - min) / divider;
                }
            }

            return result;
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (var index = 1; index < Layers.Count; ++index)
            {
                var previousLayerSignals = Layers[index - 1].GetSignals();
                var layer = Layers[index];

                layer.FeedForwardNeurons(previousLayerSignals);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new Neuron[Topology.InputCount];
            for (var index = 0; index < Topology.InputCount; ++index)
            {
                var neuron = new InputNeuron();
                inputNeurons[index] = neuron;
            }

            var inputLayer = new NeuralLayer(inputNeurons);
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayers()
        {
            foreach (var neuronsCount in Topology.NeuronsInHiddenLayersCounts)
            {
                Neuron[] hiddenNeurons = new Neuron[neuronsCount];
                var lastLayer = Layers.Last();

                for (int index = 0; index < neuronsCount; ++index)
                {
                    var neuron = new HiddenNeuron(lastLayer.Neurons.Count());
                    hiddenNeurons[index] = neuron;
                }

                var hiddenLayer = new NeuralLayer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new Neuron[Topology.OutputCount];
            var lastLayer = Layers.Last();

            for (var index = 0; index < Topology.OutputCount; ++index)
            {
                var neuron = new OutputNeuron(lastLayer.Neurons.Count());
                outputNeurons[index] = neuron;
            }

            var outputLayer = new NeuralLayer(outputNeurons);
            Layers.Add(outputLayer);
        }

        public Neuron FeedForward(IReadOnlyList<double> inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            return Topology.OutputCount == 1
                ? Layers.Last().Neurons.First()
                : Layers.Last().Neurons.OrderByDescending(x => x.Output).First();
        }

        private double BackPropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;
            var difference = actual - expected;

            foreach(var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            } 

            for (int i = Layers.Count - 2; i >= 0; --i)
            {
                var layer = Layers[i];
                var previousLayer = Layers[i + 1];

                layer.LearnNeurons(previousLayer,Topology.LearningRate);
            }

            var result = difference * difference;
            return result;
        }

        public double Learn(IReadOnlyList<double> expected, double[,] inputs, int epoch)
        {
            double[,] scaledInputs = Scale(inputs);

            var error = 0.0;
            for (int i = 0; i < epoch; ++i)
            {
                for (int row = 0; row < expected.Count(); ++row)
                {
                    var output = expected[row];
                    var input = Matrix.GetRow(scaledInputs, row);

                    error += BackPropagation(output, input);
                }
            }

            return error / epoch;
        }

        private void SendSignalsToInputNeurons(IReadOnlyList<double> inputSignals)
        {
            if (Topology.InputCount != inputSignals.Count)
                throw new Exception("The number of input signals should be equal to the number of neurons in the primary layer.");

            for (var index = 0; index < inputSignals.Count; ++index)
            {
                var signal = inputSignals[index];
                var neuron = Layers.First()[index];

                neuron.FeedForward(new double[1] { signal });
            }
        }

        public override string ToString() => $"Input: {Topology.InputCount} | Hidden: {Topology.NeuronsInHiddenLayersCount} | Output: {Topology.OutputCount} ";
    }
}
