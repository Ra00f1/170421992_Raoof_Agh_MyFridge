package com.example.my_fridge_android.ui.login

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.my_fridge_android.data.source.remote.LoginRequest
import com.example.my_fridge_android.domain.repository.MainRepository
import com.example.my_fridge_android.ui.login.LoginContract.UiAction
import com.example.my_fridge_android.ui.login.LoginContract.UiEffect
import com.example.my_fridge_android.ui.login.LoginContract.UiState
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
class LoginViewModel @Inject constructor(
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
                        usernameErrorMessage = "",
                        isLoginEnabled = validateLoginForm(
                            uiAction.username,
                            _uiState.value.password
                        )
                    )
                }
            }

            is UiAction.PasswordChanged -> {
                updateUiState {
                    copy(
                        password = uiAction.password,
                        isPasswordError = false,
                        passwordErrorMessage = "",
                        isLoginEnabled = validateLoginForm(
                            _uiState.value.username,
                            uiAction.password
                        )
                    )
                }
            }

            is UiAction.TogglePasswordVisibility -> {
                updateUiState { copy(isPasswordVisible = !isPasswordVisible) }
            }

            is UiAction.LoginClicked -> {
                println("DEBUG: LoginClicked action received")
                performLogin()
            }

            is UiAction.ForgotPasswordClicked -> {
                viewModelScope.launch {
                    emitUiEffect(UiEffect.NavigateToForgotPassword)
                }
            }

            is UiAction.SignUpClicked -> {
                viewModelScope.launch {
                    emitUiEffect(UiEffect.NavigateToSignUp)
                }
            }
        }
    }

    private fun validateLoginForm(username: String, password: String): Boolean {
        return username.isNotBlank() &&
                password.isNotBlank()
    }

    private fun performLogin() {
        val currentState = _uiState.value
        println("DEBUG: performLogin() called with username: ${currentState.username}")

        // Validate inputs
        var hasErrors = false

        if (currentState.username.isBlank()) {
            updateUiState {
                copy(
                    isUsernameError = true,
                    usernameErrorMessage = "Username is required"
                )
            }
            hasErrors = true
        }

        if (currentState.password.isBlank()) {
            updateUiState {
                copy(
                    isPasswordError = true,
                    passwordErrorMessage = "Password is required"
                )
            }
            hasErrors = true
        }

        if (hasErrors) {
            println("DEBUG: Form validation failed")
            return
        }

        // Perform actual login API call
        viewModelScope.launch {
            updateUiState { copy(isLoading = true) }

            try {
                val loginRequest = LoginRequest(
                    username = currentState.username,
                    password = currentState.password
                )

                println("DEBUG: LoginRequest created: $loginRequest")
                println("DEBUG: Calling repository.loginUser()")

                repository.loginUser(loginRequest)
                    .onSuccess { response ->
                        println("DEBUG: Login successful: $response")
                        // Show the response in a popup first
                        emitUiEffect(UiEffect.ShowLoginResponse(response))
                        // Then navigate to home after a delay or user dismisses popup
                        emitUiEffect(UiEffect.NavigateToHome)
                    }
                    .onFailure { error ->
                        println("DEBUG: Login failed: ${error.message}")
                        emitUiEffect(UiEffect.ShowError("Login failed: ${error.message}"))
                    }
            } catch (e: Exception) {
                println("DEBUG: Login exception: ${e.message}")
                emitUiEffect(UiEffect.ShowError("Login failed: ${e.message}"))
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
