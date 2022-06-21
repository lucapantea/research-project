"""Classes to represent the interest dynamics of documents and users."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
import numpy as np
from absl import flags
from gym import spaces

from recsim import choice_model
from recsim import document
from recsim import user
from recsim import utils
from recsim.simulator import environment
from recsim.simulator import recsim_gym

FLAGS = flags.FLAGS


class PDDocument(document.AbstractDocument):
    """Class to represent a dynamic user preference document."""

    MAX_LENGTH = 20.0  # Max length is 20 units
    NUM_FEATURES = 20  # Number of features for the topic vector

    def __init__(self,
                 doc_id,
                 features,
                 length=None,
                 quality=None,
                 topic_id=None):
        """Initializes a PD document."""
        super(PDDocument, self).__init__(doc_id)
        self.features = features
        self.length = length
        self.quality = quality
        self.topic_id = topic_id

    def get_dominant_topic(self):
        return np.argmax(self.features)

    def create_observation(self):
        return self.features

    @classmethod
    def observation_space(cls):
        return spaces.Box(
            shape=(cls.NUM_FEATURES,), dtype=np.float32, low=-1.0, high=1.0)


class PDDocumentSampler(document.AbstractDocumentSampler):
    """Sample a document with a new set of features."""

    def __init__(self,
                 document_ctor=PDDocument,
                 length_mean=5.0,
                 length_std=1.0,
                 **kwargs):
        """Initializes a new document sampler."""
        super(PDDocumentSampler, self).__init__(doc_ctor=document_ctor, **kwargs)
        self._document_id = 0  # keeping track of document id
        self.length_mean = length_mean
        self.length_std = length_std

    def sample_document(self):
        features = dict()
        features['doc_id'] = self._document_id
        feature_vector = self._rng.normal(0, 0.5, self.get_doc_ctor().NUM_FEATURES)
        features['features'] = np.clip(feature_vector, -1.0, 1.0)
        features['length'] = min(
            self._rng.normal(self.length_mean, self.length_std),
            self.get_doc_ctor().MAX_LENGTH)
        features['topic_id'] = np.argmax(feature_vector)
        features['quality'] = self._rng.beta(4, 2)
        self._document_id += 1
        return self._doc_ctor(**features)


class PDUserModel(user.AbstractUserModel):
    """Class to model a user with time-based dynamic preferences."""

    def __init__(self,
                 slate_size,
                 user_state_ctor=None,
                 choice_model_ctor=None,
                 response_model_ctor=None,
                 no_click_mass=1.0,
                 interest_update_rate=0.001,
                 interest_update_prob=0.005,
                 stationary=False,
                 seed=0):
        """
        Args:
            slate_size: Int representing the size of the slate
            user_state_ctor: A constructor to create the user state
            choice_model_ctor: A constructor to create the user choice model
            response_model_ctor: A constructor function to create response. The
                function should take a string of doc ID as input and returns a
                PDResponse object.
            seed: The seed used in the random sampling
        """
        if not response_model_ctor:
            raise Exception('response_model_ctor is a required callable.')
        if not choice_model_ctor:
            raise Exception('choice_model_ctor is a required callable.')
        super(PDUserModel, self).__init__(
            response_model_ctor,
            PDUserSampler(user_ctor=user_state_ctor,
                          seed=seed),
            slate_size)
        self._user_state_ctor = user_state_ctor
        self.choice_model = choice_model_ctor({'min_normalizer': self._user_state.min_normaliser,
                                               'no_click_mass': self._user_state.no_choice_penalty})

        self.no_click_mass = no_click_mass
        self.interest_update_prob = interest_update_prob
        self.interest_update_rate = interest_update_rate
        self.stationary = stationary

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return self._user_state.time_budget <= 0

    def update_state(self, slate_documents, responses):
        """Updates the user's latent state based on responses to the slate.

        Assumes response choice per slate (i.e. User selects only one document or no item).
        """
        user_state: PDUserState = self._user_state

        # Assumes a normal distribution function for the interests
        def compute_interest_gradient(interest, mean, std):
            pdf = 1 / np.sqrt(2 * np.pi * std ** 2) * np.e ** (-1 / 2 * ((interest - mean) / std) ** 2)
            gradient = -pdf * (interest - mean) / (std ** 2)
            return gradient

        for doc, res in zip(slate_documents, responses):
            if res.clicked:
                self.choice_model.score_documents(
                    user_state, [doc.create_observation()])
                # scores is a list of length 1 since only one doc observation is set.
                expected_utility = self.choice_model.scores[0]

                # Calculating user's satisfaction
                new_satisfaction = user_state.memory_discount * user_state.satisfaction + \
                                   user_state.immediate_discount * self.compute_similarity(user_state.user_interests, doc.create_observation())+\
                                   np.random.normal(0.0, user_state.noise_std)

                # successful 3
                user_state.satisfaction = 1/(1 + np.exp(-3 * new_satisfaction))

                # If the environment is non-stationary, i.e. user preferences don't shift over time
                if not self.stationary:
                    # Create update mask. For now, change 3 interest values (size)
                    num_updates = 3
                    mask = np.zeros(shape=len(doc.features))
                    mask[np.random.choice(len(doc.features), size=num_updates, replace=False)] = 1

                    # Targets for updating
                    target_interests = user_state.user_interests * mask
                    target_topics = doc.features * mask

                    # Compute the interest gradient (where the user's interests tend to go) and update value
                    interest_gradient = compute_interest_gradient(target_interests, user_state.interest_mean,
                                                                  user_state.interest_std)
                    update = self.interest_update_rate * np.absolute(interest_gradient * target_topics)

                    if np.random.rand(1) < self.interest_update_prob:
                        user_state.user_interests = np.clip(user_state.user_interests, -1.0, 1.0)
                        positive_update_prob = np.dot((np.array(user_state.user_interests) + 1.0) / (1 + 1),
                                                      mask)/num_updates
                        if np.random.rand(1) < positive_update_prob:
                            user_state.user_interests += update
                        else:
                            user_state.user_interests -= update
                    user_state.user_interests = np.clip(user_state.user_interests, -1.0, 1.0)

                received_utility = user_state.user_quality_factor * expected_utility + \
                                   user_state.document_quality_factor * doc.quality
                retention_factor = np.random.uniform(0.1, 0.2)
                user_state.time_budget -= res.engagement
                user_state.time_budget += retention_factor * res.engagement * received_utility
                return

            # Plotting purposes
            user_state.user_interests = np.clip(user_state.user_interests, -1.0, 1.0)
            # res.user_interests = np.clip(user_state.user_interests, -1.0, 1.0)

        # Step penalty if no selection
        user_state.time_budget -= user_state.no_choice_penalty

    @staticmethod
    def compute_similarity(vector_a, vector_b):
        return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

    def simulate_response(self, documents):
        """Simulates the user's response to a slate of documents with choice model."""
        # Get the list of responses
        responses = [self._response_model_ctor() for _ in documents]

        doc_obs = [doc.create_observation() for doc in documents]
        self.choice_model.score_documents(self._user_state, doc_obs)
        selected_index = self.choice_model.choose_item()

        for i, res in enumerate(responses):
            res.quality = documents[i].quality
            res.topic_id = documents[i].get_dominant_topic()

        # User made no selection
        if selected_index is None:
            return responses

        self._generate_response(documents[selected_index],
                                responses[selected_index])
        return responses

    def _generate_response(self, doc, response):
        user_state: PDUserState = self._user_state

        response.clicked = True
        topic_vector = doc.features
        interest_vector = user_state.user_interests
        similarity = self.compute_similarity(topic_vector, interest_vector)

        # Todo: figure out the engagement: maybe extract parameter (sensitivity)
        # successful 8
        # more successful 7
        watch_time = 1/(1 + np.exp(-7 * similarity)) * doc.length
        response.engagement = min(user_state.time_budget, watch_time)
        # Plotting purposes
        response.satisfaction = user_state.satisfaction
        # response.user_interests = user_state.user_interests


class PDUserState(user.AbstractUserState):
    """Class to represent users with preference dynamics."""

    # Number of features used in the user state representation.
    # used primarily for the interest vector over the topics
    NUM_FEATURES = 20

    def __init__(self,
                 interest_mean=None,
                 interest_std=None,
                 memory_discount=None,
                 immediate_discount=None,
                 noise_std=None,
                 time_budget=None,
                 no_choice_penalty=None,
                 user_quality_factor=None,
                 document_quality_factor=None,
                 min_normaliser=None):
        """Initializes a new user."""
        # State variables
        self.interest_mean = interest_mean
        self.interest_std = interest_std
        user_interest_vector = np.random.normal(self.interest_mean, self.interest_std,
                                                self.NUM_FEATURES)
        self.user_interests = np.clip(user_interest_vector, -1.0, 1.0)
        self.time_budget = time_budget

        # Transition model & satisfaction parameters
        self.memory_discount = memory_discount
        self.immediate_discount = immediate_discount
        self.noise_std = noise_std

        # Transition model parameters
        self.no_choice_penalty = no_choice_penalty
        self.user_quality_factor = user_quality_factor
        self.document_quality_factor = document_quality_factor

        # Initial (noisy) satisfaction
        self.satisfaction = np.random.normal(0.0, self.noise_std)

        # Normaliser for the user preferences (interest vector)
        self.min_normaliser = min_normaliser

    def score_document(self, observations):
        if self.user_interests.shape != observations.shape:
            raise ValueError('User and document feature dimension mismatch!')
        dominant_topic = np.zeros(shape=len(observations))
        dominant_topic[np.argmax(observations)] = 1
        return np.dot(self.user_interests, dominant_topic)

    def create_observation(self):
        return self.user_interests

    @classmethod
    def observation_space(cls):
        return spaces.Box(
            shape=(cls.NUM_FEATURES,), dtype=np.float32, low=-1.0, high=1.0)


@gin.configurable
class PDUserSampler(user.AbstractUserSampler):
    """Sample a user with a new set of features."""

    def __init__(self,
                 user_ctor=PDUserState,
                 interest_mean=0.0,
                 interest_std=0.5,
                 memory_discount=0.7,
                 immediate_discount=1.0,
                 noise_std=0.03,
                 time_budget=500, # todo: change here back to 200
                 no_choice_penalty=1.0,
                 min_normaliser=-1.0,
                 **kwargs):
        """Creates a new user state sampler."""
        self.interest_mean = interest_mean
        self.interest_std = interest_std
        self.memory_discount = memory_discount
        self.immediate_discount = immediate_discount
        self.noise_std = noise_std
        self.time_budget = time_budget
        self.no_choice_penalty = no_choice_penalty
        self.min_normaliser = min_normaliser
        super(PDUserSampler, self).__init__(user_ctor=user_ctor, **kwargs)

    def sample_user(self):
        features = dict()
        features['interest_mean'] = self.interest_mean
        features['interest_std'] = self.interest_std
        features['memory_discount'] = self.memory_discount
        features['immediate_discount'] = self.immediate_discount
        features['noise_std'] = self.noise_std
        features['time_budget'] = self.time_budget
        features['no_choice_penalty'] = self.no_choice_penalty
        features['user_quality_factor'] = self._rng.beta(9, 3)
        features['document_quality_factor'] = self._rng.beta(9, 3)
        features['min_normaliser'] = self.min_normaliser
        return self._user_ctor(**features)


##### Eval
@gin.configurable
class PDEvalUserSampler(user.AbstractUserSampler):
    """Sample a user with a new set of features for evaluation."""

    def __init__(self,
                 user_ctor=PDUserState,
                 interest_mean=0.0,  # 0.5
                 interest_std=0.5,  # 0.7
                 memory_discount=0.7,  # 0.8
                 immediate_discount=0.9,  # 4.0
                 noise_std=0.03,  # 0.1 -> user is more undecided
                 time_budget=100,  # 60
                 no_choice_penalty=1.0,
                 min_normaliser=-1.0,
                 **kwargs):
        """Creates a new user state sampler."""
        self.interest_mean = interest_mean
        self.interest_std = interest_std
        self.memory_discount = memory_discount
        self.immediate_discount = immediate_discount
        self.noise_std = noise_std
        self.time_budget = time_budget
        self.no_choice_penalty = no_choice_penalty
        self.min_normaliser = min_normaliser
        super(PDEvalUserSampler, self).__init__(user_ctor=user_ctor, **kwargs)

    def sample_user(self):
        features = dict()
        features['interest_mean'] = self.interest_mean
        features['interest_std'] = self.interest_std
        features['memory_discount'] = self.memory_discount
        features['immediate_discount'] = self.immediate_discount
        features['noise_std'] = self.noise_std
        features['time_budget'] = self.time_budget
        features['no_choice_penalty'] = self.no_choice_penalty
        features['user_quality_factor'] = self._rng.beta(9, 3)
        features['document_quality_factor'] = self._rng.beta(9, 3)
        features['min_normaliser'] = self.min_normaliser
        return self._user_ctor(**features)


class PDResponse(user.AbstractResponse):
    """Class representing a user's response to a document."""

    MIN_QUALITY = -1
    MAX_QUALITY = 1

    def __init__(self,
                 clicked=False,
                 engagement=0.0,
                 quality=0.0,
                 topic_id=0.0,
                 satisfaction=0.0,
                 user_interests=[]):
        self.clicked = clicked
        self.engagement = engagement
        self.quality = quality
        self.topic_id = topic_id

        # For plotting purposes, the Agent doesn't directly observe the satisfaction
        self.satisfaction = satisfaction
        self.user_interests = user_interests

    @staticmethod
    def response_space():
        """
        clicked: 0 (False), 1 (True)
        engagement: [0, PDDocument.MAX_LENGTH]
        topic: [0, PDDocument.NUM_FEATURES]
        quality: [MIN_QUALITY, MAX_QUALITY]
        """
        return spaces.Dict({
            'clicked': spaces.Discrete(2),
            'engagement': spaces.Box(shape=tuple(), dtype=np.float32, low=0, high=PDDocument.MAX_LENGTH),
            'quality': spaces.Box(shape=tuple(), dtype=np.float32, low=PDResponse.MIN_QUALITY,
                                  high=PDResponse.MAX_QUALITY),
            'topic_id': spaces.Discrete(PDDocument.NUM_FEATURES),
            'satisfaction': spaces.Box(shape=tuple(), dtype=np.float32, low=0, high=PDDocument.MAX_LENGTH),
            # 'user_interests': spaces.Box(shape=(PDDocument.NUM_FEATURES,), dtype=np.float32, low=-1.0, high=1.0)
        })

    def create_observation(self):
        return {
            'clicked': int(self.clicked),
            'engagement': np.array(self.engagement),
            'quality': np.array(self.quality),
            'topic_id': int(self.topic_id),
            'satisfaction': np.array(self.satisfaction),
            # 'user_interests': self.user_interests
        }


def engagement_reward(responses):
    """Calculates the total clicked engagement from a list of responses."""
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.engagement
    if isinstance(reward, np.ndarray):
        return reward[0]
    return reward


def create_environment(env_config):
    """Creates a Preference Dynamics environment."""

    user_model = PDUserModel(
        slate_size=env_config['slate_size'],
        choice_model_ctor=choice_model.MultinomialProportionalChoiceModel,
        response_model_ctor=PDResponse,
        user_state_ctor=PDUserState,
        interest_update_rate=env_config['interest_update_rate'],
        interest_update_prob=env_config['interest_update_prob'],
        stationary=env_config['stationary'],
        seed=env_config['seed'])

    document_sampler = PDDocumentSampler(
        document_ctor=PDDocument,
        length_mean=5.0,
        length_std=1.0,
        seed=env_config['seed'])

    env = environment.Environment(
        user_model=user_model,
        document_sampler=document_sampler,
        num_candidates=env_config['num_candidates'],
        slate_size=env_config['slate_size'],
        resample_documents=env_config['resample_documents'])

    return recsim_gym.RecSimGymEnv(
        raw_environment=env,
        reward_aggregator=engagement_reward,
        metrics_aggregator=utils.aggregate_pd_metrics,
        metrics_writer=utils.write_pd_metrics)
