import abc


class Classifier(metaclass=abc.ABCMeta):
    """
    Abstract Classifier.
    """
    @abc.abstractmethod
    def classify(self, img: str) -> int:
        """
        classify an image file
        :param img: file path. full absolute path recommended.
        :return: class of image.
        """
        pass
