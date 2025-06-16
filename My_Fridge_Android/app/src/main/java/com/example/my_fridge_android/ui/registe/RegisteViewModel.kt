package com.example.my_fridge_android.ui.registe

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.my_fridge_android.data.source.remote.RegisterRequest
import com.example.my_fridge_android.domain.repository.MainRepository
import com.example.my_fridge_android.ui.registe.RegisteContract.UiAction
import com.example.my_fridge_android.ui.registe.RegisteContract.UiEffect
import com.example.my_fridge_android.ui.registe.RegisteContract.UiState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class RegisteViewModel @Inject constructor(
    private val repository: MainRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _uiEffect by lazy { Channel<UiEffect>() }
    val uiEffect: Flow<UiEffect> by lazy { _uiEffect.receiveAsFlow() }

    fun onAction(uiAction: UiAction) {
        when (uiAction) {
            is UiAction.UsernameChanged -> {
                updateUiState {
                    copy(
                        username = uiAction.username,
                        isUsernameError = false,
                        usernameErrorMessage = ""
                    )
                }
                validateForm()
            }

            is UiAction.EmailChanged -> {
                updateUiState {
                    copy(
                        email = uiAction.email,
                        isEmailError = false,
                        emailErrorMessage = ""
                    )
                }
                validateForm()
            }

            is UiAction.PasswordChanged -> {
                updateUiState {
                    copy(
                        password = uiAction.password,
                        isPasswordError = false,
                        passwordErrorMessage = ""
                    )
                }
                validatePasswordMatch()
                validateForm()
            }

            is UiAction.ConfirmPasswordChanged -> {
                updateUiState {
                    copy(
                        confirmPassword = uiAction.confirmPassword,
                        isConfirmPasswordError = false,
                        confirmPasswordErrorMessage = ""
                    )
                }
                validatePasswordMatch()
                validateForm()
            }

            is UiAction.AgeChanged -> {
                updateUiState {
                    copy(
                        age = uiAction.age,
                        isAgeError = false,
                        ageErrorMessage = ""
                    )
                }
                validateForm()
            }

            is UiAction.HeightChanged -> {
                updateUiState {
                    copy(
                        height = uiAction.height,
                        isHeightError = false,
                        heightErrorMessage = ""
                    )
                }
                validateForm()
            }

            is UiAction.WeightChanged -> {
                updateUiState {
                    copy(
                        weight = uiAction.weight,
                        isWeightError = false,
                        weightErrorMessage = ""
                    )
                }
                validateForm()
            }

            is UiAction.GenderSelected -> {
                updateUiState {
                    copy(
                        gender = uiAction.gender,
                        showGenderDropdown = false
                    )
                }
            }

            is UiAction.ToggleGenderDropdown -> {
                updateUiState {
                    copy(showGenderDropdown = !showGenderDropdown)
                }
            }

            is UiAction.TogglePasswordVisibility -> {
                updateUiState {
                    copy(isPasswordVisible = !isPasswordVisible)
                }
            }

            is UiAction.ToggleConfirmPasswordVisibility -> {
                updateUiState {
                    copy(isConfirmPasswordVisible = !isConfirmPasswordVisible)
                }
            }

            is UiAction.RegisterClicked -> {
                println("DEBUG: RegisterClicked action received")
                println("DEBUG: Form validation result: ${validateAllFields()}")
                if (validateAllFields()) {
                    println("DEBUG: Starting registration process")
                    performRegistration()
                } else {
                    println("DEBUG: Form validation failed")
                }
            }

            is UiAction.BackToLoginClicked -> {
                viewModelScope.launch {
                    emitUiEffect(UiEffect.NavigateToLogin)
                }
            }
        }
    }

    private fun validateForm() {
        val currentState = _uiState.value
        val isFormValid = currentState.username.isNotBlank() &&
                currentState.email.isNotBlank() &&
                isValidEmail(currentState.email) &&
                currentState.password.isNotBlank() &&
                currentState.password.length >= 6 &&
                currentState.confirmPassword.isNotBlank() &&
                currentState.password == currentState.confirmPassword &&
                currentState.age.isNotBlank() &&
                isValidAge(currentState.age) &&
                currentState.height.isNotBlank() &&
                isValidHeight(currentState.height) &&
                currentState.weight.isNotBlank() &&
                isValidWeight(currentState.weight) &&
                !currentState.isUsernameError &&
                !currentState.isEmailError &&
                !currentState.isPasswordError &&
                !currentState.isConfirmPasswordError &&
                !currentState.isAgeError &&
                !currentState.isHeightError &&
                !currentState.isWeightError

        updateUiState {
            copy(isRegisterEnabled = isFormValid)
        }
    }

    private fun validatePasswordMatch() {
        val currentState = _uiState.value
        if (currentState.confirmPassword.isNotBlank() &&
            currentState.password != currentState.confirmPassword
        ) {
            updateUiState {
                copy(
                    isConfirmPasswordError = true,
                    confirmPasswordErrorMessage = "Passwords do not match"
                )
            }
        }
    }

    private fun validateAllFields(): Boolean {
        val currentState = _uiState.value
        var isValid = true

        // Validate username
        if (currentState.username.isBlank()) {
            updateUiState {
                copy(
                    isUsernameError = true,
                    usernameErrorMessage = "Username is required"
                )
            }
            isValid = false
        } else if (currentState.username.length < 3) {
            updateUiState {
                copy(
                    isUsernameError = true,
                    usernameErrorMessage = "Username must be at least 3 characters"
                )
            }
            isValid = false
        }

        // Validate email
        if (currentState.email.isBlank()) {
            updateUiState {
                copy(
                    isEmailError = true,
                    emailErrorMessage = "Email is required"
                )
            }
            isValid = false
        } else if (!isValidEmail(currentState.email)) {
            updateUiState {
                copy(
                    isEmailError = true,
                    emailErrorMessage = "Please enter a valid email"
                )
            }
            isValid = false
        }

        // Validate password
        if (currentState.password.isBlank()) {
            updateUiState {
                copy(
                    isPasswordError = true,
                    passwordErrorMessage = "Password is required"
                )
            }
            isValid = false
        } else if (currentState.password.length < 6) {
            updateUiState {
                copy(
                    isPasswordError = true,
                    passwordErrorMessage = "Password must be at least 6 characters"
                )
            }
            isValid = false
        }

        // Validate confirm password
        if (currentState.confirmPassword.isBlank()) {
            updateUiState {
                copy(
                    isConfirmPasswordError = true,
                    confirmPasswordErrorMessage = "Please confirm your password"
                )
            }
            isValid = false
        } else if (currentState.password != currentState.confirmPassword) {
            updateUiState {
                copy(
                    isConfirmPasswordError = true,
                    confirmPasswordErrorMessage = "Passwords do not match"
                )
            }
            isValid = false
        }

        // Validate age
        if (currentState.age.isBlank()) {
            updateUiState {
                copy(
                    isAgeError = true,
                    ageErrorMessage = "Age is required"
                )
            }
            isValid = false
        } else if (!isValidAge(currentState.age)) {
            updateUiState {
                copy(
                    isAgeError = true,
                    ageErrorMessage = "Age must be between 13 and 120"
                )
            }
            isValid = false
        }

        // Validate height
        if (currentState.height.isBlank()) {
            updateUiState {
                copy(
                    isHeightError = true,
                    heightErrorMessage = "Height is required"
                )
            }
            isValid = false
        } else if (!isValidHeight(currentState.height)) {
            updateUiState {
                copy(
                    isHeightError = true,
                    heightErrorMessage = "Height must be between 100 and 250 cm"
                )
            }
            isValid = false
        }

        // Validate weight
        if (currentState.weight.isBlank()) {
            updateUiState {
                copy(
                    isWeightError = true,
                    weightErrorMessage = "Weight is required"
                )
            }
            isValid = false
        } else if (!isValidWeight(currentState.weight)) {
            updateUiState {
                copy(
                    isWeightError = true,
                    weightErrorMessage = "Weight must be between 30 and 300 kg"
                )
            }
            isValid = false
        }

        return isValid
    }

    private fun isValidEmail(email: String): Boolean {
        return android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()
    }

    private fun isValidAge(age: String): Boolean {
        return try {
            val ageInt = age.toInt()
            ageInt in 13..120
        } catch (e: NumberFormatException) {
            false
        }
    }

    private fun isValidHeight(height: String): Boolean {
        return try {
            val heightInt = height.toInt()
            heightInt in 100..250
        } catch (e: NumberFormatException) {
            false
        }
    }

    private fun isValidWeight(weight: String): Boolean {
        return try {
            val weightInt = weight.toInt()
            weightInt in 30..300
        } catch (e: NumberFormatException) {
            false
        }
    }

    private fun performRegistration() {
        println("DEBUG: performRegistration() called")
        updateUiState { copy(isLoading = true) }

        viewModelScope.launch {
            val currentState = _uiState.value
            println("DEBUG: Current state - username: ${currentState.username}, email: ${currentState.email}")

            try {
                val registerRequest = RegisterRequest(
                    username = currentState.username,
                    password = currentState.password,
                    email = currentState.email,
                    age = currentState.age.toInt(),
                    height = currentState.height.toInt(),
                    weight = currentState.weight.toInt(),
                    gender = currentState.gender
                )

                println("DEBUG: RegisterRequest created: $registerRequest")
                println("DEBUG: Calling repository.registerUser()")

                repository.registerUser(registerRequest)
                    .onSuccess { message ->
                        println("DEBUG: Registration successful: $message")
                        emitUiEffect(UiEffect.NavigateToLogin)
                    }
                    .onFailure { error ->
                        println("DEBUG: Registration failed: ${error.message}")
                        emitUiEffect(UiEffect.ShowError("Registration failed: ${error.message}"))
                    }
            } catch (e: Exception) {
                println("DEBUG: Registration exception: ${e.message}")
                emitUiEffect(UiEffect.ShowError("Registration failed: ${e.message}"))
            } finally {
                updateUiState { copy(isLoading = false) }
            }
        }
    }

    private fun updateUiState(block: UiState.() -> UiState) {
        _uiState.update(block)
    }

    private suspend fun emitUiEffect(uiEffect: UiEffect) {
        _uiEffect.send(uiEffect)
    }
}
