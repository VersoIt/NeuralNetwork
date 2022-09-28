using System.Globalization;
using Neuron.Tools;

namespace Neuron
{
    public abstract class Neuron
    {
        protected Neuron(int inputCount, NeuronTypes type = NeuronTypes.Normal)
        {
            Weights = new double[inputCount];
            NeuronType = type;
        }

        public virtual void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Length)
                throw new Exception("Can't create current neuron!");

            Array.Copy(weights, Weights, weights.Length);
        }

        public double Delta { get; protected set; }

        public double[] Weights { get; init; }

        public NeuronTypes NeuronType { get; init; }

        public double Output { get; protected set; }

        public abstract void Learn(double error, double learRate);

        public abstract double FeedForward(double[] inputs);

        public override string ToString() => Output.ToString(CultureInfo.InvariantCulture);
    }
}
