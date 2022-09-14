using Neuron.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuron
{
    public class DefaultNeuron : Neuron
    {
        public DefaultNeuron(int inputCount, NeuronTypes type = NeuronTypes.Normal)
            : base(inputCount, type)
        {
            Weights = new double[inputCount];
        }

        public override void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Length)
                throw new Exception("Невозможно создать такой нейрон!");

            Array.Copy(weights, Weights, weights.Length);
        }

        public override double[] Weights { get; init; }

        public override NeuronTypes NeuronType { get; init; }

        public override double Output { get; protected set; }

        public override double FeedForward(double[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new Exception("Число входных данных нейрона не равно числу входов в нейрон!");

            double sum = 0;

            for (int index = 0; index < Weights.Length; ++index)
                sum += inputs[index] * Weights[index];

            Output = new Sigmoida().GetResultBy(sum);
            return Output;
        }

        public override string ToString() => Output.ToString();
    }
}
