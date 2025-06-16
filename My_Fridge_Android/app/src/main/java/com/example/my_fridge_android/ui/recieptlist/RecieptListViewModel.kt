package com.example.my_fridge_android.ui.recieptlist

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.my_fridge_android.data.config.UserPreferences
import com.example.my_fridge_android.data.source.remote.FridgeItem
import com.example.my_fridge_android.domain.repository.MainRepository
import com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiAction
import com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiEffect
import com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*
import javax.inject.Inject

@HiltViewModel
class RecieptListViewModel @Inject constructor(
    private val repository: MainRepository,
    private val userPreferences: UserPreferences
) : ViewModel() {

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _uiEffect by lazy { Channel<UiEffect>() }
    val uiEffect: Flow<UiEffect> by lazy { _uiEffect.receiveAsFlow() }

    init {
        loadFridgeItems()
    }

    fun onAction(uiAction: UiAction) {
        when (uiAction) {
            is UiAction.NavigateToHome -> {
                viewModelScope.launch {
                    emitUiEffect(UiEffect.NavigateToHome)
                }
            }

            is UiAction.ShowImagePicker -> {
                updateUiState { copy(showImagePickerSheet = true) }
            }

            is UiAction.HideImagePicker -> {
                updateUiState { copy(showImagePickerSheet = false) }
            }

            is UiAction.TakePhoto -> {
                // Prevent duplicate camera launches
                if (_uiState.value.isCameraLaunching) {
                    return
                }
                updateUiState { 
                    copy(
                        showImagePickerSheet = false,
                        isCameraLaunching = true
                    ) 
                }
                viewModelScope.launch {
                    emitUiEffect(UiEffect.OpenCamera)
                    // Add timeout to reset camera launching state
                    kotlinx.coroutines.delay(10000) // 10 seconds timeout
                    if (_uiState.value.isCameraLaunching) {
                        updateUiState { copy(isCameraLaunching = false) }
                        println("Camera launch timeout - resetting state")
                    }
                }
            }

            is UiAction.SelectFromGallery -> {
                updateUiState { copy(showImagePickerSheet = false) }
                viewModelScope.launch {
                    emitUiEffect(UiEffect.OpenGallery)
                }
            }

            is UiAction.TestServerConnection -> {
                testServerConnection()
            }

            is UiAction.ClearUploadStatus -> {
                updateUiState { copy(uploadStatus = null) }
            }

            is UiAction.DeleteReceipt -> {
                println("DEBUG: DeleteReceipt action triggered with receiptId: ${uiAction.receiptId}")
                deleteReceipt(uiAction.receiptId)
            }

            is UiAction.LoadFridgeItems -> {
                loadFridgeItems()
            }

            is UiAction.ShowAddIngredients -> {
                updateUiState { copy(showAddIngredientsSheet = true) }
            }

            is UiAction.HideAddIngredients -> {
                updateUiState { copy(showAddIngredientsSheet = false) }
            }

            is UiAction.AddIngredient -> {
                addIngredient(uiAction.name, uiAction.amount, uiAction.amountType)
            }

            is UiAction.IncreaseItemAmount -> {
                increaseItemAmount(uiAction.itemId)
            }

            is UiAction.DecreaseItemAmount -> {
                decreaseItemAmount(uiAction.itemId)
            }
        }
    }

    fun uploadImage(imageUri: Uri) {
        viewModelScope.launch {
            updateUiState {
                copy(
                    isLoading = true,
                    uploadStatus = "Connecting to server..."
                )
            }

            // First ping the server to test connection
            repository.pingServer()
                .onSuccess { response ->
                    println("Ping successful: $response")
                    updateUiState { copy(uploadStatus = "Server connected! Uploading image...") }
                    // If ping succeeds, proceed with upload
                    uploadImageToServer(imageUri)
                }
                .onFailure { error ->
                    println("Ping failed: ${error.message}")
                    updateUiState {
                        copy(
                            isLoading = false,
                            uploadStatus = "❌ Connection failed: ${error.message}"
                        )
                    }
                }
            }
        }

    private suspend fun uploadImageToServer(imageUri: Uri) {
        repository.uploadReceiptImage(imageUri)
            .onSuccess { message ->
                println("Upload successful: $message")
                updateUiState {
                    copy(
                        isLoading = false,
                        uploadStatus = "✅ Upload successful: $message"
                    )
                }
                // Refresh the list
                loadFridgeItems()
            }
            .onFailure { error ->
                println("Upload failed: ${error.message}")
                updateUiState {
                    copy(
                        isLoading = false,
                        uploadStatus = "❌ Upload failed: ${error.message}"
                    )
                }
            }
    }

    fun testServerConnection() {
        viewModelScope.launch {
            updateUiState {
                copy(
                    isLoading = true,
                    uploadStatus = "Testing server connection..."
                )
            }

            repository.pingServer()
                .onSuccess { response ->
                    println("Server connection test successful: $response")
                    updateUiState {
                        copy(
                            isLoading = false,
                            uploadStatus = "✅ Server test successful: $response"
                        )
                    }
                }
                .onFailure { error ->
                    println("Server connection test failed: ${error.message}")
                    updateUiState {
                        copy(
                            isLoading = false,
                            uploadStatus = "❌ Server test failed: ${error.message}"
                        )
                    }
                }
        }
    }

    private fun loadFridgeItems() {
        viewModelScope.launch {
            updateUiState { copy(isLoading = true) }

            val userId = userPreferences.getUserId()
            if (userId == -1) {
                updateUiState {
                    copy(
                        isLoading = false,
                        errorMessage = "User not logged in"
                    )
                }
                emitUiEffect(UiEffect.ShowError("User not logged in"))
                return@launch
            }

            repository.getFridgeItems(userId)
                .onSuccess { items ->
                    updateUiState {
                        copy(
                            isLoading = false,
                            fridgeItems = items,
                            errorMessage = null
                        )
                    }
                }
                .onFailure { error ->
                    updateUiState {
                        copy(
                            isLoading = false,
                            errorMessage = error.message
                        )
                    }
                    emitUiEffect(UiEffect.ShowError(error.message ?: "Failed to load fridge items"))
                }
        }
    }

    private fun deleteReceipt(receiptId: String) {
        println("DEBUG: deleteReceipt called with receiptId: $receiptId")

        // Set deleting state
        updateUiState {
            copy(isDeleting = true, deletingItemId = receiptId)
        }

        viewModelScope.launch {
            try {
                println("DEBUG: Inside viewModelScope.launch")
                val userId = userPreferences.getUserId()
                println("DEBUG: Retrieved userId: $userId")
                if (userId == -1) {
                    println("DEBUG: User not logged in, showing error")
                    updateUiState { copy(isDeleting = false, deletingItemId = null) }
                    emitUiEffect(UiEffect.ShowError("User not logged in"))
                    return@launch
                }

                // Find the item to delete
                val currentItems = _uiState.value.fridgeItems
                val itemToDelete = currentItems.find { it.getSafeId().toString() == receiptId }
                println("DEBUG: Found item to delete: $itemToDelete")
                if (itemToDelete == null) {
                    println("DEBUG: Item not found, showing error")
                    updateUiState { copy(isDeleting = false, deletingItemId = null) }
                    emitUiEffect(UiEffect.ShowError("Item not found"))
                    return@launch
                }

                val productName = itemToDelete.getSafeProduct().getSafeName()
                println("DEBUG: About to call deleteProduct API with userId=$userId, productName=$productName")

                // Call the delete API
                repository.deleteProduct(userId, productName)
                    .onSuccess { response ->
                        println("DEBUG: Product deleted successfully: $response")
                        // Remove from local state after successful API call
                        updateUiState {
                            copy(
                                fridgeItems = fridgeItems.filter {
                                    it.getSafeId().toString() != receiptId
                                },
                                isDeleting = false,
                                deletingItemId = null
                            )
                        }
                        println("DEBUG: Updated UI state, item should be removed from list")
                        // Show success message
                        emitUiEffect(UiEffect.ShowMessage("Ingredient deleted successfully"))
                    }
                    .onFailure { error ->
                        println("DEBUG: Failed to delete product: ${error.message}")
                        // Reset deleting state
                        updateUiState { copy(isDeleting = false, deletingItemId = null) }
                        // Show error message with details
                        emitUiEffect(UiEffect.ShowError("Failed to delete ingredient: ${error.message}"))
                    }
            } catch (e: Exception) {
                println("DEBUG: Exception in deleteReceipt: ${e.message}")
                e.printStackTrace()
                updateUiState { copy(isDeleting = false, deletingItemId = null) }
                emitUiEffect(UiEffect.ShowError("Error deleting item: ${e.message}"))
            }
        }
    }

    private fun addIngredient(name: String, amount: String, amountType: String) {
        viewModelScope.launch {
            updateUiState { copy(showAddIngredientsSheet = false) }

            val userId = userPreferences.getUserId()
            if (userId == -1) {
                emitUiEffect(UiEffect.ShowError("User not logged in"))
                return@launch
            }

            // Show loading state
            updateUiState { copy(isLoading = true) }

            try {
                repository.addManualIngredient(userId, name, amount, amountType)
                    .onSuccess { response ->
                        println("DEBUG: Manual ingredient added successfully: $response")
                        updateUiState { copy(isLoading = false) }
                        emitUiEffect(UiEffect.ShowMessage("Ingredient added successfully"))
                        // Refresh the fridge items list
                        loadFridgeItems()
                    }
                    .onFailure { error ->
                        println("DEBUG: Failed to add manual ingredient: ${error.message}")
                        updateUiState { copy(isLoading = false) }
                        emitUiEffect(UiEffect.ShowError("Failed to add ingredient: ${error.message}"))
                    }
            } catch (e: Exception) {
                println("DEBUG: Exception in addIngredient: ${e.message}")
                e.printStackTrace()
                updateUiState { copy(isLoading = false) }
                emitUiEffect(UiEffect.ShowError("Error adding ingredient: ${e.message}"))
            }
        }
    }

    private fun decreaseItemAmount(itemId: Int) {
        viewModelScope.launch {
            try {
                val userId = userPreferences.getUserId()
                if (userId == -1) {
                    emitUiEffect(UiEffect.ShowError("User not logged in"))
                    return@launch
                }

                // Find the item to decrease
                val currentItems = _uiState.value.fridgeItems
                val itemToDecrease = currentItems.find { it.getSafeId() == itemId }
                if (itemToDecrease == null) {
                    emitUiEffect(UiEffect.ShowError("Item not found"))
                    return@launch
                }

                // Calculate smart decrement amount
                val decrementAmount =
                    getSmartIncrementAmount(itemToDecrease.getSafeProduct().getSafeCountType())

                // Check if item would have negative amount
                if (itemToDecrease.getSafeAmount() <= decrementAmount) {
                    emitUiEffect(UiEffect.ShowError("Cannot reduce below minimum amount"))
                    return@launch
                }

                // Set updating state
                updateUiState {
                    copy(
                        isDeleting = true,
                        deletingItemId = itemId.toString()
                    )
                }

                // Call the reduce amount API
                repository.reduceAmount(
                    userId = userId,
                    productName = itemToDecrease.getSafeProduct().getSafeName(),
                    amount = decrementAmount,
                    amountType = itemToDecrease.getSafeProduct().getSafeCountType()
                )
                    .onSuccess { response ->
                        println("DEBUG: Amount reduced successfully: $response")
                        // Update local state after successful API call
                        updateUiState {
                            copy(
                                fridgeItems = fridgeItems.map { item ->
                                    if (item.getSafeId() == itemId) {
                                        item.copy(amount = item.getSafeAmount() - decrementAmount)
                                    } else {
                                        item
                                    }
                                },
                                isDeleting = false,
                                deletingItemId = null
                            )
                        }
                        emitUiEffect(UiEffect.ShowMessage("Amount reduced successfully"))
                    }
                    .onFailure { error ->
                        println("DEBUG: Failed to reduce amount: ${error.message}")
                        updateUiState { copy(isDeleting = false, deletingItemId = null) }
                        emitUiEffect(UiEffect.ShowError("Failed to reduce amount: ${error.message}"))
                    }
            } catch (e: Exception) {
                println("DEBUG: Exception in decreaseItemAmount: ${e.message}")
                e.printStackTrace()
                updateUiState { copy(isDeleting = false, deletingItemId = null) }
                emitUiEffect(UiEffect.ShowError("Error reducing amount: ${e.message}"))
            }
        }
    }

    private fun increaseItemAmount(itemId: Int) {
        viewModelScope.launch {
            try {
                val userId = userPreferences.getUserId()
                if (userId == -1) {
                    emitUiEffect(UiEffect.ShowError("User not logged in"))
                    return@launch
                }

                // Find the item to increase
                val currentItems = _uiState.value.fridgeItems
                val itemToIncrease = currentItems.find { it.getSafeId() == itemId }
                if (itemToIncrease == null) {
                    emitUiEffect(UiEffect.ShowError("Item not found"))
                    return@launch
                }

                // Calculate smart increment amount
                val incrementAmount =
                    getSmartIncrementAmount(itemToIncrease.getSafeProduct().getSafeCountType())

                // Set updating state
                updateUiState {
                    copy(
                        isDeleting = true,
                        deletingItemId = itemId.toString()
                    )
                }

                // Call the add amount API
                repository.addAmount(
                    userId = userId,
                    productName = itemToIncrease.getSafeProduct().getSafeName(),
                    categoryName = itemToIncrease.getSafeProduct().getSafeCategory(),
                    countType = itemToIncrease.getSafeProduct().getSafeCountType(),
                    price = itemToIncrease.getSafeProduct().getSafePrice(),
                    amount = incrementAmount,
                    skt = itemToIncrease.getSafeSkt()
                )
                    .onSuccess { response ->
                        println("DEBUG: Amount added successfully: $response")
                        // Update local state after successful API call
                        updateUiState {
                            copy(
                                fridgeItems = fridgeItems.map { item ->
                                    if (item.getSafeId() == itemId) {
                                        item.copy(amount = item.getSafeAmount() + incrementAmount)
                                    } else {
                                        item
                                    }
                                },
                                isDeleting = false,
                                deletingItemId = null
                            )
                        }
                        emitUiEffect(UiEffect.ShowMessage("Amount increased successfully"))
                    }
                    .onFailure { error ->
                        println("DEBUG: Failed to add amount: ${error.message}")
                        updateUiState { copy(isDeleting = false, deletingItemId = null) }
                        emitUiEffect(UiEffect.ShowError("Failed to increase amount: ${error.message}"))
                    }
            } catch (e: Exception) {
                println("DEBUG: Exception in increaseItemAmount: ${e.message}")
                e.printStackTrace()
                updateUiState { copy(isDeleting = false, deletingItemId = null) }
                emitUiEffect(UiEffect.ShowError("Error increasing amount: ${e.message}"))
            }
        }
    }

    /**
     * Returns smart increment/decrement amount based on unit type:
     * - L, amount, KG: change by 1
     * - G, ml: change by 100
     */
    private fun getSmartIncrementAmount(unitType: String): Int {
        return when (unitType.lowercase()) {
            "g", "ml" -> 100
            "l", "amount", "kg" -> 1
            else -> 1 // Default to 1 for unknown types
        }
    }

    private fun updateUiState(block: UiState.() -> UiState) {
        _uiState.update(block)
    }

    private suspend fun emitUiEffect(uiEffect: UiEffect) {
        _uiEffect.send(uiEffect)
    }

    fun resetCameraLaunching() {
        updateUiState { copy(isCameraLaunching = false) }
    }
}
