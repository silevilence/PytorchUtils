from utils.Classifier import Classifier

import json
import os

from typing import List


class ComposedClassifier(Classifier):
    """
    Composed classifier.
    Combine several classifiers to a classify chain.
    """

    def __init__(self, config: str = ''):
        # load configs
        default_config_file = open(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.path.pardir, 'presets', 'default_classifier.json')), 'r')
        default_config = json.load(default_config_file)
        if not (isinstance(config, str) and os.path.exists(config)):
            user_config = default_config
        else:
            user_config_file = open(config, 'r')
            user_config: list = json.load(user_config_file)

        # generate list of classifiers
        self.clas: List[dict] = list()
        for classifier_config in user_config:
            classifier_module = __import__(classifier_config['module'], fromlist=[classifier_config['classifier']])
            classifier_class = getattr(classifier_module, classifier_config['classifier'])
            classifier_app: Classifier = classifier_class(**classifier_config['params'])

            classifier = dict(classifier=classifier_app,
                              condition=classifier_config['condition'])
            self.clas.append(classifier)

        # close file stream
        if 'user_config_file' in locals() or 'user_config_file' in globals():
            user_config_file.close()
        default_config_file.close()

    def classify(self, image_path: str):
        # variable
        result: int = -1

        # classify
        for classifier in self.clas:
            need_run = eval(classifier['condition'])
            if need_run:
                result = classifier['classifier'].classify(image_path)

        return result
