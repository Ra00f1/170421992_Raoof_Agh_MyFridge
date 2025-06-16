package com.example.my_fridge_android.ui.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.my_fridge_android.data.config.UserPreferences
import com.example.my_fridge_android.domain.repository.MainRepository
import com.example.my_fridge_android.ui.home.HomeContract.UiAction
import com.example.my_fridge_android.ui.home.HomeContract.UiEffect
import com.example.my_fridge_android.ui.home.HomeContract.UiState
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
class HomeViewModel @Inject constructor(
    private val userPreferences: UserPreferences,
    private val repository: MainRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _uiEffect by lazy { Channel<UiEffect>() }
    val uiEffect: Flow<UiEffect> by lazy { _uiEffect.receiveAsFlow() }

    init {
        loadUserData()
    }

    private fun loadUserData() {
        val userId = userPreferences.getUserId()
        val username = userPreferences.getUsername()
        updateUiState {
            copy(
                userId = userId,
                userName = username
            )
        }
    }

    fun onAction(uiAction: UiAction) {
        when (uiAction) {
            is UiAction.NavigateToReceiptList -> {
                viewModelScope.launch {
                    emitUiEffect(UiEffect.NavigateToReceiptList)
                }
            }

            is UiAction.NavigateToRecipeChat -> {
                viewModelScope.launch {
                    emitUiEffect(UiEffect.NavigateToRecipeChat)
                }
            }

            is UiAction.SendItemsToRecipeAssistant -> {
                sendItemsToRecipeAssistant()
            }

            is UiAction.Logout -> {
                viewModelScope.launch {
                    emitUiEffect(UiEffect.NavigateToLogin)
                }
            }
        }
    }

    private fun updateUiState(block: UiState.() -> UiState) {
        _uiState.update(block)
    }

    private suspend fun emitUiEffect(uiEffect: UiEffect) {
        _uiEffect.send(uiEffect)
    }

    private fun sendItemsToRecipeAssistant() {
        viewModelScope.launch {
            updateUiState { copy(isSendingToRecipeAssistant = true) }

            try {
                val userId = userPreferences.getUserId()
                if (userId == -1) {
                    emitUiEffect(UiEffect.ShowMessage("User not logged in"))
                    return@launch
                }

                // First get the fridge items
                repository.getFridgeItems(userId)
                    .onSuccess { fridgeItems ->
                        // Then send them to Recipe Assistant
                        repository.sendItemsToRecipeAssistant(fridgeItems)
                            .onSuccess { response ->
                                println("DEBUG: HomeViewModel - Received API response: $response")
                                // Navigate to RecipeChat with the API response
                                emitUiEffect(UiEffect.NavigateToRecipeChatWithResponse(response))
                            }
                            .onFailure { error ->
                                emitUiEffect(UiEffect.ShowMessage("Failed to send items: ${error.message}"))
                            }
                    }
                    .onFailure { error ->
                        emitUiEffect(UiEffect.ShowMessage("Failed to get fridge items: ${error.message}"))
                    }
            } finally {
                updateUiState { copy(isSendingToRecipeAssistant = false) }
            }
        }
    }
}
