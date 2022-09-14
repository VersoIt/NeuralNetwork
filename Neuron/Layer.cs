using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuron
{
    public class Layer : IEnumerable
    {

        public Layer(Neuron[] neurons, NeuronTypes type)
        {
            // TODO: проверить все входные нейроны на соответствие одному типу
            Neurons = neurons;
            NeuronType = type;
        }

        NeuronTypes NeuronType { get; }

        public Neuron[] Neurons { get; }

        public Neuron this[int index]
        {
            get => Neurons[index];
            set => Neurons[index] = value;
        }

        public double[] GetSignals()
        {
            var signals = new double[Neurons.Count()];

            for (int index = 0; index < signals.Length; ++index)
                signals[index] = Neurons[index].Output;

            return signals;
        }

        public IEnumerator GetEnumerator() => Neurons.GetEnumerator();
    }
}
