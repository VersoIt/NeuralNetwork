using System.Drawing;

namespace PictureConverter.Abstract
{
    internal interface IConvertible<out T, in K> 
    {
        public T Convert(K target);

        public double[] ConvertIntoBinarizedArray(K target);
    }
}
