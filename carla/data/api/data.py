from abc import ABC, abstractmethod


class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user. This is the general data object
    that is used in CARLA.
    """

    @property
    @abstractmethod
    def categorical(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        pass

    @property
    @abstractmethod
    def target_train(self):
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        pass

    @property
    @abstractmethod
    def target_test(self):
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        pass

    @property
    @abstractmethod
    def df_train(self):
        """
        The training split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_test(self):
        """
        The testing split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        """
        The vocabulary size of the activity column

        Returns
        -------
        Integer
        """
        pass

    @property
    @abstractmethod
    def max_prefix_length(self):
        """
        The maximum length of the (prefix) sequences

        Returns
        -------
        Integer
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """
        The name of the dataset

        Returns
        -------
        String
        """
        pass