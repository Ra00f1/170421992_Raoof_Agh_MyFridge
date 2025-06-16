package com.example.my_fridge_android.ui.recipechat

data class Recipe(
    val title: String,
    val ingredients: List<String>,
    val instructions: List<String>,
    val cookingTime: String = "",
    val difficulty: String = ""
)

data class ChatMessage(
    val id: String,
    val content: String,
    val isFromUser: Boolean,
    val timestamp: Long = System.currentTimeMillis(),
    val recipes: List<Recipe> = emptyList(),
    val hasOptions: Boolean = false,
    val options: List<String> = emptyList()
)

object RecipeChatContract {
    data class UiState(
        val isLoading: Boolean = false,
        val messages: List<ChatMessage> = emptyList(),
        val currentMessage: String = "",
        val isMessageSending: Boolean = false,
        val errorMessage: String = "",
        val isInitialResponse: Boolean = false,
        val initialApiResponse: String = "",
        val waitingForOptionSelection: Boolean = false,
        val isChatDisabled: Boolean = false
    )

    sealed class UiAction {
        data class MessageChanged(val message: String) : UiAction()
        data object SendMessage : UiAction()
        data object BackToHome : UiAction()
        data object ClearError : UiAction()
        data class InitializeWithApiResponse(val response: String) : UiAction()
        data class SelectOption(val optionNumber: String) : UiAction()
    }

    sealed class UiEffect {
        data object NavigateToHome : UiEffect()
        data class ShowError(val message: String) : UiEffect()
    }
}
