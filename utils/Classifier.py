import abc


class Classifier(metaclass=abc.ABCMeta):
    """
    Abstract Classifier.
    """
    @abc.abstractmethod
    def classify(self, image_path: str) -> int:
        """
        classify an image file
        :param image_path: file path. full absolute path recommended.
        :return: class of image.
        """
        pass
