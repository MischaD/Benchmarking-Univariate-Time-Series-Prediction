import numpy as np
from copy import deepcopy


class CNNValidityChecker:
    """
    Test if input for cnn is good enough for the output to be dependant on the entire input sequence i.e. every
    output node sees every input node.
    """
    def __init__(self,
                 N,
                 dilation_factors,
                 kernel_size,
                 sequence_length,
                 causal_conv_only=False):
        """
        :param N: Number of stacked blocks
        :param dilation_factors: dilation factors of the layers in each block
        :param kernel_size: conv1d kernel size. Interpreted as symmetrical. Has to be odd
        :param sequence_length: input/output sequence lenght of the model
        :param causal_conv_only: only compute causal convolutions. If set to true the kernel size has to be adapted accordingly (kernel_size_causal = (kernel_size + 1 // 2)
        """
        assert kernel_size % 2 != 0
        self.N = N
        self.dilation_factors = dilation_factors
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.causal_conv_only = causal_conv_only
        self._history = []

        # calculate receptive field via some sort of manual 'backpropagation' through the network
        output = self.start_backpropoagation()
        self.is_valid = all(output == 1)

    @property
    def history(self):
        return np.array(self._history)

    def start_backpropoagation(self):
        """
        check if the last node sees every input node. If it does it is easy to show that the same goes for all other
        output nodes

        :return: array of length sequence length with a 1 at every node that is reached by the last output node.
        """
        input = np.zeros(self.sequence_length)
        input[-1] = 1
        self._history.append(deepcopy(input))
        for i in range(self.N):
            input = self.backprop_through_block(input)
        return input

    def backprop_through_block(self, input):
        """Compute every node reached by every activated input node for one block

        :param input: array of length sequence length with 1 at every activated node
        :return: array where newly activated nodes have now the value 1
        """
        for dil in reversed(self.dilation_factors):
            input = self.backprop_conv_layer(input, dil)
        return input

    def backprop_conv_layer(self, input, dilation):
        """Compute every node reached by every activated input node for one conv layer

        :param input: array of length sequence length with 1 at every activated node
        :param dilation: dilation of the conv layer
        :return: array where newly activated nodes have now the value 1
        """
        # indices of all nodes that are activated
        activated = np.arange(self.sequence_length)[input == 1]

        # calculate relative indices that will be reached after the current layer
        kernel_reach = (self.kernel_size + 1) // 2
        dilated_kernel_reach = [dilation * i for i in range(kernel_reach)]
        for i in activated:
            for idx in dilated_kernel_reach:
                if i - idx < 0:
                    pass
                else:
                    input[i - idx] = 1

                if self.causal_conv_only:
                    continue

                if i + idx >= self.sequence_length:
                    pass
                else:
                    input[i + idx] = 1

        self._history.append(deepcopy(input))
        return input
