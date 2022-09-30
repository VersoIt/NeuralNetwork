using System.Globalization;

namespace NeuralNetwork.Abstract
{
    public abstract class Neuron
    {
        protected Neuron(int inputCount)
        {
            Weights = new double[inputCount];
        }

        public abstract void Learn(double error, double learningRate);

        public virtual void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Length)
                throw new Exception("Can't create current neuron!");

            Array.Copy(weights, Weights, weights.Length);
        }

        public abstract string NeuronType { get; }

        public double Delta { get; protected set; }

        public double[] Weights { get; init; }

        public double Output { get; protected set; }

        public abstract double FeedForward(IReadOnlyList<double> inputs);

        public override string ToString() => Output.ToString(CultureInfo.InvariantCulture);
    }
}
