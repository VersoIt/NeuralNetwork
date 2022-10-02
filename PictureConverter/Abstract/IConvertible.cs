
namespace PictureConverter.Abstract
{
    internal interface IConvertible<out T>
    {
        public T Convert();
    }
}
