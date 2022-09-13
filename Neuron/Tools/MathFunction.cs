using System.Collections.Generic;
using System.Collections;
using System.Linq;

namespace Neuron.Tools
{
    public abstract class MathFunction
    {
        public abstract double GetResultBy(double x);

        public override string ToString() => $"{{ {string.Join(" ", Enumerable.Range(-10, 10).Select(x => GetResultBy(x)))} }}";
    }
}
