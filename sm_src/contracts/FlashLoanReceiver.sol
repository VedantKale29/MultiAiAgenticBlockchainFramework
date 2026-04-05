// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * FlashLoanReceiver.sol
 * ======================
 * STAGE 2 — Victim contract (holds ETH, gets drained by attacker via pool fee)
 *
 * PURPOSE:
 *   Represents a DeFi protocol that has ETH in the pool.
 *   The NaiveReceiverLenderPool charges fees to this contract,
 *   draining its ETH balance.
 */

contract FlashLoanReceiver {

    address public pool;
    address public owner;

    event ReceivedFlashLoan(uint256 amount, uint256 fee);

    constructor(address _pool) payable {
        pool  = _pool;
        owner = msg.sender;
    }

    modifier onlyPool() {
        require(msg.sender == pool, "Only pool");
        _;
    }

    /**
     * Called by the pool during flash loan execution.
     * Must repay amount + fee.
     */
    function receiveETH(uint256 fee) external payable onlyPool {
        emit ReceivedFlashLoan(msg.value, fee);
        // Repay the pool (amount + fee)
        (bool ok, ) = pool.call{value: msg.value + fee}("");
        require(ok, "Repayment failed");
    }

    // Fund this contract
    receive() external payable {}

    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function withdraw() external {
        require(msg.sender == owner, "Not owner");
        payable(owner).transfer(address(this).balance);
    }
}
