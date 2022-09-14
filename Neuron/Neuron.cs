using Neuron.Tools;

namespace Neuron
{
    public abstract class Neuron
    {
        public Neuron(int inputCount, NeuronTypes type = NeuronTypes.Normal)
        {
            Weights = new double[inputCount];
            NeuronType = type;
        }

        public virtual void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Length)
                throw new Exception("Невозможно создать такой нейрон!");

            Array.Copy(weights, Weights, weights.Length);
        }

        public abstract double[] Weights { get; init; }

        public abstract NeuronTypes NeuronType { get; init; }

        public abstract double Output { get; protected set; }

        public abstract double FeedForward(double[] inputs);

        public override string ToString() => Output.ToString();
    }
}
