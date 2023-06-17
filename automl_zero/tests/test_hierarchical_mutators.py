import numpy as np
from automl_zero.hierarchical.mutators import _mutate_all, mutate_winner

def test_mutate_all():
    # Set the seed for random number generation
    np.random.seed(0)

    # Arrange
    winner = np.zeros((3, 6))  # Assuming the winner is a 2D array. Adjust as needed.
    gene_to_mutate = "gene_setup"
    memory_dict_len = 5
    MAX_ARG = 2
    
    # Act
    mutated = _mutate_all(winner, gene_to_mutate, memory_dict_len, MAX_ARG)

    # Assert
    # Note: the exact assertions will depend on what you expect the function to do.
    # Here are some examples of possible assertions.
    assert mutated.shape == winner.shape
    assert not np.array_equal(winner, mutated)  # Verify that the function has changed the winner.
    #assert np.max(mutated[:, 0:1]) <= NUMBER_OF_META_OPS
    for op_value, metalevel in zip(mutated[:, 0:1], mutated[:, 1:2]):
        if metalevel > 0:
            assert op_value <= NUMBER_OF_META_OPS
    assert np.max(mutated[:, 1:2]) <= METALEVEL_COUNT
    assert np.max(mutated[:, 2:2+MAX_ARG]) <= memory_dict_len
    assert np.max(mutated[:, 2+MAX_ARG:3+MAX_ARG]) <= memory_dict_len
    assert np.max(mutated[:, 3+MAX_ARG:]) <= 100  # Assuming the constants are in the range [-100, 100].

# Similarly for _mutate_add_or_remove_one_instruction and _mutate_one_argument.

def test_mutate_winner():
    # Set the seed for random number generation
    np.random.seed(0)

    # Arrange
    winner_metagene = {
        "gene_setup": np.zeros((3, 6)),  # Assuming the genes are 2D arrays. Adjust as needed.
        "gene_pred": np.zeros((3, 6)),
        "gene_learn": np.zeros((3, 6))
    }
    memory_dict_len = 5
    
    # Act
    new_metagene = mutate_winner(winner_metagene, memory_dict_len)

    # Assert
    # Note: the exact assertions will depend on what you expect the function to do.
    assert isinstance(new_metagene, dict)
    assert len(new_metagene) == len(winner_metagene)
    for key in winner_metagene.keys():
        assert not np.array_equal(winner_metagene[key], new_metagene[key])  # Verify that the function has mutated at least one gene.

if __name__ == "__main__":
    test_mutate_all()
    test_mutate_winner()