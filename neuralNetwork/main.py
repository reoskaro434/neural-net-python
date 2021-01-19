import random
import math


class Neuron():
    learning_speed = 0.03

    def __init__(self, weight_amount):
        self.weight_list = []
        self.last_output = None
        self.last_error = None

        for x in range(weight_amount):
            self.weight_list.insert(x, random.uniform(0, 1))

    @staticmethod
    def __activation_function(argument):
        return math.tanh(argument)

    @staticmethod
    def __derivative_of_activation_function(argument):
        return 1 - pow(math.tanh(argument), 2)

    def calculate_first_layer_output(self, argument):
        sum_tmp = 0
        for x in range(len(argument)):
            sum_tmp += argument[x] * self.weight_list[x]
        self.last_output = self.__activation_function(sum_tmp)

    def calculate_output(self, left_layer):
        sum_tmp = 0
        for x in range(len(self.weight_list)):
            sum_tmp += (Neuron.return_last_output(left_layer[x]) * self.weight_list[x])
        self.last_output = self.__activation_function(sum_tmp)

    def calculate_last_layer_error(self, expected_value):
        self.last_error = self.__derivative_of_activation_function(self.last_output) * (
                    expected_value - self.last_output)

    def calculate_error(self, right_layer, neuron_index):
        error_sum = 0
        for x in range(len(right_layer)):
            error_sum += self.__derivative_of_activation_function(
                self.last_output) * Neuron.return_weight_multiplied_by_error(right_layer[x], neuron_index)
        self.last_error = error_sum

    def return_weight_multiplied_by_error(self, weight_index):
        return self.weight_list[weight_index] * self.last_error

    def weight_correction(self, left_layer):
        for x in range(len(self.weight_list)):
            self.weight_list[
                x] += Neuron.return_last_output(
                left_layer[x]) * self.learning_speed * self.last_error

    def weight_correction_first_layer(self, argument_vector):
        for x in range(len(self.weight_list)):
            self.weight_list[
                x] += argument_vector[x] * self.learning_speed * self.last_error

    def return_last_output(self):
        return self.last_output

    def return_last_error(self):
        return self.last_error


class Net():
    net_schematic = []
    neuron_list = []

    def __init__(self, schematic):
        self.net_schematic = schematic
        self.__create_net()

    def __create_net(self):
        for x in range(len(self.net_schematic)):
            temporary_list = []
            for y in range(self.net_schematic[x]):
                if x == 0:
                    temporary_list.insert(y, Neuron(self.net_schematic[x]))
                else:
                    temporary_list.insert(y, Neuron(self.net_schematic[x - 1]))
            self.neuron_list.insert(x, temporary_list)

    def __propagate_the_signal(self, argument_vector):
        for x in range(len(self.neuron_list)):
            for y in range(len(self.neuron_list[x])):
                if x == 0:
                    Neuron.calculate_first_layer_output(self.neuron_list[x][y], argument_vector)
                else:
                    Neuron.calculate_output(self.neuron_list[x][y], self.neuron_list[x - 1])

    def __calculate_errors(self, expected_value_vector):
        for x in range((len(self.neuron_list) - 1), -1, -1):
            for y in range(len(self.neuron_list[x])):
                if x == (len(self.neuron_list) - 1):
                    Neuron.calculate_last_layer_error(self.neuron_list[x][y], expected_value_vector[y])
                else:
                    Neuron.calculate_error(self.neuron_list[x][y], self.neuron_list[x + 1], y)

    def __weight_corrections(self, argument_vector):
        for x in range(len(self.neuron_list)):
            for y in range(len(self.neuron_list[x])):
                if x == 0:
                    Neuron.weight_correction_first_layer(self.neuron_list[x][y], argument_vector)
                else:
                    Neuron.weight_correction(self.neuron_list[x][y], self.neuron_list[x - 1])

    def __calculate_final_error(self):
        pass

    def learn(self, data_list, expected_values):
        expected_val = [0]
        for z in range(6000):
            for x in range(len(data_list)):
                argument_vector = []
                for y in range(len(data_list[x])):
                    argument_vector.insert(y, data_list[x][y])
                    expected_val[0] = expected_values[x]

                self.__propagate_the_signal(argument_vector)

                self.__calculate_errors(expected_val)

                self.__weight_corrections(argument_vector)
                if z % 100 == True:
                    print("main output is : ", round(Neuron.return_last_output(self.neuron_list[1][0]), 6),
                          " should be: ",
                          expected_val[0])
            if z % 100 == True:
                print("")


if __name__ == '__main__':
    myNet = Net([3, 1])
    a = 1  # bias
    data = [[a, 0, 0], [a, 1, 1], [a, 0, 1], [a, 1, 0]]
    predictions = [1, 1, 0, 0]

    print(" START ")

    myNet.learn(data, predictions)
