using System.Runtime.CompilerServices;
using NeuralNetwork.Abstract;

namespace NeuralNetwork
{
    internal class InputNeuron : Neuron
    {
        public InputNeuron()
            : base(1)
        {
            Weights = new double[1];
            Inputs = new double[1];

            InitWeights(1);
        }

        public override string NeuronType { get; } = "Input";

        public double[] Inputs { get; }

        public override void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Length)
                throw new Exception("This is impossible to create such a input neuron!");

            Array.Copy(weights, 0, Weights, 0, weights.Length);
        }

        private void InitWeights(int inputCount)
        {
            for (var i = 0; i < inputCount; ++i)
            {
                Weights[i] = 1;                
                Inputs[i] = 0.0;
            }
        }

        public override void Learn(double error, double learningRate)
        {
        }

        public override double FeedForward(IReadOnlyList<double> inputs)
        {
            Array.Copy(inputs.ToArray(), Inputs, inputs.Count());

            if (inputs.Count() != Weights.Length)
                throw new Exception("The number of input signals is not equal to the number of inputs to the input neuron!");

            var sum = Weights.Select((t, index) => inputs[index] * t).Sum();

            Output = sum;

            return Output;
        }
    }
}
