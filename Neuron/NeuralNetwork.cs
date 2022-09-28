using System.Data;
using Neuron.Tools;


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

        private double[,] Scale(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

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

        private double[,] Normalize(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

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
                    result[row, column] = (inputs[row, column] - average) / standardError;
                }
            }

            return result;
        }

        public Topology Topology { get; }

        public List<Layer> Layers { get; }

        private void CreateInputLayer()
        {
            var inputNeurons = new Neuron[Topology.InputCount];
            for (var index = 0; index < Topology.InputCount; ++index)
            {
                var neuron = new InputNeuron();
                inputNeurons[index] = neuron;
            }

            var inputLayer = new Layer(inputNeurons);
            Layers.Add(inputLayer);
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

            var outputLayer = new Layer(outputNeurons);
            Layers.Add(outputLayer);
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

                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        public Neuron FeedForward(double[] inputSignals)
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

            foreach(var item in Layers.Last().Neurons)
            {
                if (item is ILearnable neuron)
                    neuron.Learn(difference, Topology.LearningRate);
            } 

            for (int i = Layers.Count - 2; i >= 0; --i)
            {
                var layer = Layers[i];
                var previousLayer = Layers[i + 1];

                for (int j = 0; j < layer.Neurons.Count(); ++j)
                {
                    var neuron = layer.Neurons[j];
                    for (int k = 0; k < previousLayer.Neurons.Count(); ++k)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[j] * previousNeuron.Delta;

                        if (neuron is ILearnable learnNeuron)
                            learnNeuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            var result = difference * difference;
            return result;
        }

        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var signals = Scale(inputs);

            var error = 0.0;
            for (int i = 0; i < epoch; ++i)
            {
                for (int row = 0; row < expected.Length; ++row)
                {
                    var output = expected[row];
                    var input = Matrix.GetRow(signals, row);

                    error += BackPropagation(output, input);
                }
            }

            return error / epoch;
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (var index = 1; index < Layers.Count; ++index)
            {
                var previousLayerSignals = Layers[index - 1].GetSignals();
                var layer = Layers[index];

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
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
    }
}
