using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuron
{
    public struct Topology
    {

        public Topology(int inputCount, int outputCount, params int[] neuronsInHiddenLayersCounts)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            NeuronsInHiddenLayersCounts = new int[neuronsInHiddenLayersCounts.Length];

            neuronsInHiddenLayersCounts.CopyTo(NeuronsInHiddenLayersCounts, 0);
        }

        public Int32 InputCount { get; }
        public Int32 OutputCount { get; }
        public int[] NeuronsInHiddenLayersCounts { get; }


    }
}
