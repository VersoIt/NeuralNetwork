using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuron.Tools
{
    internal class Sigmoida : MathFunction
    {
        public override double GetResultBy(double x) => 1.0 / (1.0 - Math.Pow(Math.E, -x));
    }
}
